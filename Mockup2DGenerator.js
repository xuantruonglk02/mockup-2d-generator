/**
 * @typedef {{name: string, image_path: string}} MockupImageLayer
 *
 * @typedef {{name: string, color_mask_path: string, is_color_image: boolean}} MockupColorLayer
 *
 * @typedef {{
 *  name: string,
 *  side: string,
 *  warp_type: string,
 *  warp_info: {artwork_width: number, artwork_height: number, model_json: string},
 *  mask_path: string,
 * }} MockupDesignLayer
 *
 * @typedef {{name: string, blend_mode: string, image_path: string}} MockupBlendLayer
 *
 * @typedef {MockupImageLayer | MockupColorLayer | MockupDesignLayer | MockupBlendLayer} MockupLayer
 *
 * @typedef {{
 *  name: string,
 *  side: string,
 *  size: {width: number, height: number},
 *  type: string,
 *  parts: MockupLayer[]
 * }} MockupInfo
 */

class Mockup2DGenerator {
    constructor() {
        this.renderCanvas = new OffscreenCanvas(1000, 1000)
        this.initWebGL(this.renderCanvas)

        // Initialize caches
        this.imageCache = new Map()
        this.textureCache = new Map()
        this.bufferCache = new Map()
        this.warpDataCache = new Map()
        this.maskTextureCache = new Map()

        // Pre-allocated buffers
        this.sharedPositionBuffer = null
        this.sharedTexCoordBuffer = null
        this.fullScreenQuadData = null

        this.maxImageCacheSize = 50
    }

    initWebGL() {
        const startTs = Date.now()

        const gl = this.renderCanvas.getContext('webgl', {
            premultipliedAlpha: true,
            preserveDrawingBuffer: true,
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance',
        })
        if (!gl) {
            throw new Error('WebGL is not supported')
        }

        // Check for required extensions
        const extensions = ['OES_texture_float', 'OES_texture_float_linear', 'WEBGL_compressed_texture_s3tc']
        extensions.forEach((ext) => {
            if (!gl.getExtension(ext)) {
                console.warn(`WebGL extension ${ext} not available`)
            }
        })

        // Vertex shader
        const vsSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            varying vec2 v_maskCoord;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;                    // Texture coordinates for the warped image
                v_maskCoord = (a_position + 1.0) / 2.0;     // Convert clip space [-1, 1] to texture space [0, 1]
                v_maskCoord.y = 1.0 - v_maskCoord.y;
            }
        `

        // Fragment shader
        const fsSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            varying vec2 v_maskCoord;
            uniform sampler2D u_texture;
            uniform sampler2D u_mask;
            uniform bool u_useMask;
    
            void main() {
                vec4 color = texture2D(u_texture, v_texCoord);
                if (u_useMask) {
                    vec4 mask = texture2D(u_mask, v_maskCoord);
                    float maskValue = mask.a;
                    color *= maskValue;
                }
                gl_FragColor = color;
            }
        `

        const vertexShader = this._compileShader(gl, gl.VERTEX_SHADER, vsSource)
        const fragmentShader = this._compileShader(gl, gl.FRAGMENT_SHADER, fsSource)

        const program = gl.createProgram()
        gl.attachShader(program, vertexShader)
        gl.attachShader(program, fragmentShader)
        gl.linkProgram(program)
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error('Failed to create program: ' + gl.getProgramInfoLog(program))
        }

        gl.useProgram(program)

        gl.enable(gl.BLEND)
        gl.clearColor(0, 0, 0, 0)

        // Cache attribute and uniform locations
        this.locations = {
            aPosition: gl.getAttribLocation(program, 'a_position'),
            aTexCoord: gl.getAttribLocation(program, 'a_texCoord'),
            uTexture: gl.getUniformLocation(program, 'u_texture'),
            uMask: gl.getUniformLocation(program, 'u_mask'),
            uUseMask: gl.getUniformLocation(program, 'u_useMask'),
        }

        this.gl = gl
        this.program = program

        // Pre-create shared buffers
        this._initializeSharedBuffers()

        // console.debug(`[Mockup2DGenerator] initWebGL | ${Date.now() - startTs} ms`)
    }

    _initializeSharedBuffers() {
        // Full screen quad data
        this.fullScreenQuadData = {
            positions: new Float32Array([
                -1.0, -1.0, // Bottom-left
                1.0, -1.0, // Bottom-right
                -1.0, 1.0, // Top-left
                1.0, 1.0, // Top-right
            ]),
            texCoords: new Float32Array([
                0.0, 1.0, // Bottom-left
                1.0, 1.0, // Bottom-right
                0.0, 0.0, // Top-left
                1.0, 0.0, // Top-right
            ]),
        }

        // Create shared position buffer
        this.sharedPositionBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sharedPositionBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.fullScreenQuadData.positions, this.gl.STATIC_DRAW)

        // Create shared texcoord buffer
        this.sharedTexCoordBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sharedTexCoordBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.fullScreenQuadData.texCoords, this.gl.STATIC_DRAW)
    }

    /**
     *
     * @param {MockupInfo} mockupInfo
     */
    async loadMockupInfos(mockupInfos) {
        let startTs = Date.now()

        // console.debug(`[Mockup2DGenerator] loadMockupInfos`, mockupInfos)
        this.mockupInfos = mockupInfos

        const mockupWidth = this.mockupInfos[0]?.size.width || 1000
        const mockupHeight = this.mockupInfos[0]?.size.height || 1000

        this.renderCanvas.width = mockupWidth
        this.renderCanvas.height = mockupHeight
        this.gl.viewport(0, 0, mockupWidth, mockupHeight)

        // Pre-cache static layers
        await Promise.all(mockupInfos.map((m) => this._preCacheLayers(m)))

        // console.debug(`[Mockup2DGenerator] loadMockupInfo | ${Date.now() - startTs} ms`)
    }

    async _preCacheLayers(mockupInfo) {
        const layers = mockupInfo.parts || []
        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        // Pre-load all static images in parallel
        const imagePromises = []

        for (const layer of layers) {
            if (layer.image_path) {
                imagePromises.push(this._loadImage(layer.image_path, canvasSize))
            }
            if (layer.color_mask_path) {
                imagePromises.push(this._loadImage(layer.color_mask_path, canvasSize))
            }
            if (layer.mask_path) {
                imagePromises.push(this._loadImage(layer.mask_path, canvasSize))
            }
            if (layer.warp_info?.model_json) {
                imagePromises.push(this._loadWarpData(layer.warp_info.model_json))
            }
        }
        if (mockupInfo.heather_mapping) {
            Object.values(mockupInfo.heather_mapping).forEach((url) => {
                imagePromises.push(this._loadImage(url, null, true))
            })
        }

        await Promise.all(imagePromises)
    }

    async render(design, color, heatherPath, useBackground = false) {
        const startTs = Date.now()

        /** @type {MockupLayer[]} */
        const mockupInfos = this.mockupInfos || []
        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        const artwork = await this._resizeArtworkCanvas(design, canvasSize)

        const mockups = []
        for (const mockupInfo of mockupInfos) {
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT)
            await this._renderMockupInfo(mockupInfo, artwork, color, heatherPath, useBackground)
            const mockupBlobURL = await this.exportAsBlobURL()
            mockups.push(mockupBlobURL)
        }

        const ctx = artwork.getContext('2d')
        ctx?.clearRect(0, 0, artwork.width, artwork.height)

        console.debug(`[Mockup2DGenerator] render | ${Date.now() - startTs} ms`)
        
        return mockups
    }

    async _renderMockupInfo(mockupInfo, artwork, color, heatherPath, useBackground = false) {
        /** @type {MockupLayer[]} */
        const layers = mockupInfo?.parts || []

        const heatherMapping = mockupInfo.heather_mapping || {}

        // Then render other layers in order
        for (const layer of layers) {
            let startTs = Date.now()

            try {
                if (!useBackground && layer.name === `${mockupInfo.name}.BG`) continue

                if (layer.color_mask_path && layer.is_color_image) {
                    await this.glFillColor(layer, color)

                    if (heatherPath) {
                        if (heatherPath === 'default') {
                            await this.glDrawHeatherLayer(layer, heatherMapping['heather_texture'])
                        } else {
                            await this.glDrawHeatherLayer(layer, heatherPath)
                        }
                    }
                } else if (layer.warp_type === 'warp_npy' && layer.warp_info) {
                    await this.glWarpAndDrawDesign(layer, artwork)
                } else if (layer.blend_mode || layer.image_path) {
                    await this.glDrawImageLayer(layer)
                } else {
                    console.warn(`[Mockup2DGenerator] Unsupported layer:`, layer)
                }
            } catch (error) {
                console.error('[Mockup2DGenerator] Error during rendering:', error)
            }

            console.debug(`[Mockup2DGenerator] _renderMockupInfo ${mockupInfo.name} "${layer.name}" | ${Date.now() - startTs} ms`)
        }
    }

    /**
     *
     * @param {MockupImageLayer | MockupBlendLayer} layer
     */
    async glDrawImageLayer(layer) {
        const blendMode = layer.blend_mode || 'normal'
        this._glSetBlendMode(blendMode)

        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        const imagePath = layer.image_path
        if (!imagePath) throw new Error('[glDrawImageLayer] Image path is required')

        // Get or create texture from cache
        let texture = this.textureCache.get(imagePath)
        if (!texture) {
            const image = await this._loadImage(imagePath, canvasSize)

            // Create texture
            texture = this.gl.createTexture()
            this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, image)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE)

            this.gl.pixelStorei(this.gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true)

            this.textureCache.set(imagePath, texture)
        }

        this._setupFullScreenQuad()

        // Set texture uniform
        this.gl.uniform1i(this.locations.uTexture, 0)
        this.gl.uniform1i(this.locations.uUseMask, 0)

        this.gl.activeTexture(this.gl.TEXTURE0)
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture)

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4)
    }

    /**
     *
     * @param {MockupColorLayer} layer
     * @param {string} color
     * @param {string} [heatherTexturePath] - Optional heather texture path
     */
    async glFillColor(layer, color) {
        const {color_mask_path} = layer
        const rgb = this._hexToRgb(color)
        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        // Create color texture
        const colorKey = `color_${color}`
        let colorTexture = this.textureCache.get(colorKey)
        if (!colorTexture) {
            colorTexture = this.gl.createTexture()
            this.gl.bindTexture(this.gl.TEXTURE_2D, colorTexture)
            const pixel = new Uint8Array([rgb.r, rgb.g, rgb.b, 255])
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, 1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixel)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST)
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST)
            this.textureCache.set(colorKey, colorTexture)
        }

        // Get mask texture
        let maskTexture = this.maskTextureCache.get(color_mask_path)
        if (!maskTexture) {
            const maskImage = await this._loadImage(color_mask_path, canvasSize)
            maskTexture = this._createTexture(maskImage)
            this.maskTextureCache.set(color_mask_path, maskTexture)
        }

        this._glSetBlendMode('normal')
        this._setupFullScreenQuad()

        // Set uniforms
        this.gl.uniform1i(this.locations.uTexture, 0)
        this.gl.uniform1i(this.locations.uMask, 1)
        this.gl.uniform1i(this.locations.uUseMask, 1)

        // Bind textures
        this.gl.activeTexture(this.gl.TEXTURE0)
        this.gl.bindTexture(this.gl.TEXTURE_2D, colorTexture)
        this.gl.activeTexture(this.gl.TEXTURE1)
        this.gl.bindTexture(this.gl.TEXTURE_2D, maskTexture)

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4)
    }

    async glDrawHeatherLayer(layer, heatherTexturePath) {
        if (!heatherTexturePath) return

        const {color_mask_path} = layer
        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        // Load heather texture
        let heatherTexture = this.textureCache.get(heatherTexturePath)
        if (!heatherTexture) {
            const heatherImage = await this._loadHeatherTexture(heatherTexturePath, canvasSize)
            heatherTexture = this._createTexture(heatherImage)
            this.textureCache.set(heatherTexturePath, heatherTexture)
        }

        // Get mask texture (same as color layer)
        let maskTexture = this.maskTextureCache.get(color_mask_path)
        if (!maskTexture) {
            const maskImage = await this._loadImage(color_mask_path, canvasSize)
            maskTexture = this._createTexture(maskImage)
            this.maskTextureCache.set(color_mask_path, maskTexture)
        }

        // Use overlay/multiply blend mode for heather
        this._glSetBlendMode('multiply')
        this._setupFullScreenQuad()

        // Set uniforms
        this.gl.uniform1i(this.locations.uTexture, 0)
        this.gl.uniform1i(this.locations.uMask, 1)
        this.gl.uniform1i(this.locations.uUseMask, 1)

        // Bind textures
        this.gl.activeTexture(this.gl.TEXTURE0)
        this.gl.bindTexture(this.gl.TEXTURE_2D, heatherTexture)
        this.gl.activeTexture(this.gl.TEXTURE1)
        this.gl.bindTexture(this.gl.TEXTURE_2D, maskTexture)

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4)
    }

    /**
     *
     * @param {MockupDesignLayer} layer
     * @param {Image} artwork
     */
    async glWarpAndDrawDesign(layer, artwork) {
        const {mask_path, warp_info} = layer
        const {model_json} = warp_info

        const canvasSize = [this.renderCanvas.width, this.renderCanvas.height]

        this._glSetBlendMode('normal')

        // Get cached warp data
        const cacheKey = `${layer.name}_warp`
        let warpBuffers = this.bufferCache.get(cacheKey)

        if (!warpBuffers) {
            // Load and process warp data
            const uvData = await this._loadWarpData(model_json)
            warpBuffers = this._createWarpBuffers(uvData)
            this.bufferCache.set(cacheKey, warpBuffers)
        }

        // Bind buffers
        this.gl.enableVertexAttribArray(this.locations.aPosition)
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, warpBuffers.vertexBuffer)
        this.gl.vertexAttribPointer(this.locations.aPosition, 2, this.gl.FLOAT, false, 0, 0)

        this.gl.enableVertexAttribArray(this.locations.aTexCoord)
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, warpBuffers.texCoordBuffer)
        this.gl.vertexAttribPointer(this.locations.aTexCoord, 2, this.gl.FLOAT, false, 0, 0)

        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, warpBuffers.indexBuffer)

        // Create design texture
        const designTexture = this._createTexture(artwork)

        // Get or create mask texture from cache
        let maskTexture = this.maskTextureCache.get(mask_path)
        if (!maskTexture) {
            const maskImage = await this._loadImage(mask_path, canvasSize)
            maskTexture = this._createTexture(maskImage)
            this.maskTextureCache.set(mask_path, maskTexture)
        }

        // Set uniforms
        this.gl.uniform1i(this.locations.uTexture, 0)
        this.gl.uniform1i(this.locations.uMask, 1)
        this.gl.uniform1i(this.locations.uUseMask, 1)

        this.gl.activeTexture(this.gl.TEXTURE0)
        this.gl.bindTexture(this.gl.TEXTURE_2D, designTexture)
        this.gl.activeTexture(this.gl.TEXTURE1)
        this.gl.bindTexture(this.gl.TEXTURE_2D, maskTexture)

        // Draw mesh
        this.gl.drawElements(this.gl.TRIANGLES, warpBuffers.indexCount, this.gl.UNSIGNED_SHORT, 0)

        this.gl.deleteTexture(designTexture)
    }

    _createWarpBuffers(uvData) {
        const N = uvData.length
        const M = uvData[0].length
        const mappedPoints = uvData.flat()

        const vertices = []
        const texCoords = []
        const indices = []

        // Generate mesh data
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < M; j++) {
                const index = i * M + j

                const vertexX = (j / (M - 1)) * 2 - 1
                const vertexY = (i / (N - 1)) * 2 - 1
                vertices.push(vertexX, -vertexY)

                if (index < mappedPoints.length) {
                    let [srcX, srcY] = mappedPoints[index]
                    texCoords.push(this._clamp(srcX, 0, 1), this._clamp(srcY, 0, 1))
                } else {
                    texCoords.push(j / (M - 1), i / (N - 1))
                }
            }
        }

        // Generate indices
        for (let i = 0; i < N - 1; i++) {
            for (let j = 0; j < M - 1; j++) {
                const topLeft = i * M + j
                const topRight = i * M + j + 1
                const bottomLeft = (i + 1) * M + j
                const bottomRight = (i + 1) * M + j + 1

                indices.push(topLeft, bottomLeft, topRight)
                indices.push(topRight, bottomLeft, bottomRight)
            }
        }

        // Create buffers
        const vertexBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vertexBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(vertices), this.gl.STATIC_DRAW)

        const texCoordBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, texCoordBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(texCoords), this.gl.STATIC_DRAW)

        const indexBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer)
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), this.gl.STATIC_DRAW)

        return {
            vertexBuffer,
            texCoordBuffer,
            indexBuffer,
            indexCount: indices.length,
        }
    }

    _createTexture(image) {
        const texture = this.gl.createTexture()
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, image)
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR)
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR)
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE)
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE)
        return texture
    }

    _setupFullScreenQuad() {
        const bufferCacheKey = 'fullscreen_quad'
        let buffers = this.bufferCache.get(bufferCacheKey)

        if (!buffers) {
            const positions = new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
            const texCoords = new Float32Array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

            const positionBuffer = this.gl.createBuffer()
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer)
            this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW)

            const texCoordBuffer = this.gl.createBuffer()
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, texCoordBuffer)
            this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW)

            buffers = {positionBuffer, texCoordBuffer}
            this.bufferCache.set(bufferCacheKey, buffers)
        }

        // Bind position buffer
        this.gl.enableVertexAttribArray(this.locations.aPosition)
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffers.positionBuffer)
        this.gl.vertexAttribPointer(this.locations.aPosition, 2, this.gl.FLOAT, false, 0, 0)

        // Bind texture coordinate buffer
        this.gl.enableVertexAttribArray(this.locations.aTexCoord)
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffers.texCoordBuffer)
        this.gl.vertexAttribPointer(this.locations.aTexCoord, 2, this.gl.FLOAT, false, 0, 0)
    }

    async _loadWarpData(url) {
        if (this.warpDataCache.has(url)) {
            return this.warpDataCache.get(url)
        }

        const response = await fetch(url)
        const data = await response.json()
        return data
    }

    async exportAsBlobURL() {
        const blob = await this.renderCanvas.convertToBlob()
        if (!blob) {
            console.error('[Mockup2DGenerator] Failed to create blob from canvas')
            return null
        }
        return URL.createObjectURL(blob)
    }

    async exportAsImageBitmap() {
        return this.renderCanvas.transferToImageBitmap()
    }

    clearCache() {
        // Clear texture cache
        this.textureCache.forEach((texture) => {
            this.gl.deleteTexture(texture)
        })
        this.textureCache.clear()

        // Clear mask texture cache
        this.maskTextureCache.forEach((texture) => {
            this.gl.deleteTexture(texture)
        })
        this.maskTextureCache.clear()

        // Clear buffer cache
        this.bufferCache.forEach((buffers) => {
            if (buffers.vertexBuffer) this.gl.deleteBuffer(buffers.vertexBuffer)
            if (buffers.texCoordBuffer) this.gl.deleteBuffer(buffers.texCoordBuffer)
            if (buffers.indexBuffer) this.gl.deleteBuffer(buffers.indexBuffer)
            if (buffers.positionBuffer) this.gl.deleteBuffer(buffers.positionBuffer)
        })
        this.bufferCache.clear()

        // Clear other caches
        this.imageCache.clear()
        this.warpDataCache.clear()
    }

    destroy() {
        this.clearCache()

        // Delete shared buffers
        if (this.sharedPositionBuffer) {
            this.gl.deleteBuffer(this.sharedPositionBuffer)
        }
        if (this.sharedTexCoordBuffer) {
            this.gl.deleteBuffer(this.sharedTexCoordBuffer)
        }

        // Delete program
        if (this.program) {
            this.gl.deleteProgram(this.program)
        }

        // Lose WebGL context
        const loseContext = this.gl.getExtension('WEBGL_lose_context')
        if (loseContext) {
            loseContext.loseContext()
        }
    }

    _compileShader(gl, type, source) {
        const shader = gl.createShader(type)
        gl.shaderSource(shader, source)
        gl.compileShader(shader)

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const error = gl.getShaderInfoLog(shader)
            gl.deleteShader(shader)
            throw new Error('Failed to compile shader: ' + error)
        }

        return shader
    }

    _hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
        return result
            ? {
                  r: parseInt(result[1], 16),
                  g: parseInt(result[2], 16),
                  b: parseInt(result[3], 16),
              }
            : {r: 255, g: 255, b: 255}
    }

    async _loadImage(src, resize, useCache = true) {
        // Check cache first
        const cacheKey = `${src}_${resize ? resize.join('x') : 'original'}`
        if (useCache && this.imageCache.has(cacheKey)) {
            return this.imageCache.get(cacheKey)
        }

        // Check cache size limit
        if (this.imageCache.size >= this.maxImageCacheSize) {
            // Remove oldest entries (first ones in the Map)
            const keysToDelete = Array.from(this.imageCache.keys()).slice(0, 10)
            keysToDelete.forEach((key) => this.imageCache.delete(key))
        }

        return new Promise((resolve, reject) => {
            const img = new Image()
            img.crossOrigin = 'Anonymous'
            img.src = src
            img.onload = () => {
                if (!resize) {
                    if (useCache) this.imageCache.set(cacheKey, img)
                    return resolve(img)
                }

                // Create a canvas for resizing
                const canvas = new OffscreenCanvas(resize[0], resize[1])
                const ctx = canvas.getContext('2d')

                // Draw the resized image
                ctx.drawImage(img, 0, 0, resize[0], resize[1])

                if (useCache) this.imageCache.set(cacheKey, canvas)
                resolve(canvas)
            }
            img.onerror = (error) => reject(error)
        })
    }

    /**
     * Load and prepare heather texture, scaling or tiling as needed
     * @param {string} src - Path to heather texture
     * @param {number[]} targetSize - [width, height] to scale/tile to
     * @returns {Promise<OffscreenCanvas>}
     */
    async _loadHeatherTexture(src, targetSize) {
        const [targetWidth, targetHeight] = targetSize

        // Load the heather image
        const heatherImg = await this._loadImage(src, null, false) // Don't resize initially

        // Create canvas at target size
        const canvas = new OffscreenCanvas(targetWidth, targetHeight)
        const ctx = canvas.getContext('2d')

        const imgWidth = heatherImg.width || heatherImg.naturalWidth
        const imgHeight = heatherImg.height || heatherImg.naturalHeight

        // If heather is smaller than target, tile it; otherwise scale it
        if (imgWidth <= targetWidth && imgHeight <= targetHeight) {
            // Tile the heather pattern
            for (let y = 0; y < targetHeight; y += imgHeight) {
                for (let x = 0; x < targetWidth; x += imgWidth) {
                    ctx.drawImage(heatherImg, x, y)
                }
            }
        } else {
            // Scale down to fit
            ctx.drawImage(heatherImg, 0, 0, targetWidth, targetHeight)
        }

        return canvas
    }

    async _resizeArtworkCanvas(canvas, resize) {
        const oCanvas = new OffscreenCanvas(resize[0], resize[1])
        const ctx = oCanvas.getContext('2d')
        ctx.drawImage(canvas, 0, 0, resize[0], resize[1])
        return oCanvas
    }

    _glSetBlendMode(mode) {
        if (mode === 'multiply') {
            this.gl.blendFunc(this.gl.DST_COLOR, this.gl.ONE_MINUS_SRC_ALPHA)
        } else if (mode === 'screen') {
            this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_COLOR)
        } else if (mode === 'linear_dodge') {
            this.gl.blendFunc(this.gl.ONE, this.gl.ONE)
        } else if (mode === 'color_dodge') {
            this.gl.blendFunc(this.gl.ONE, this.gl.ONE)
        } else {
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA)
        }
    }

    _clamp(value, min, max) {
        return Math.max(min, Math.min(max, value))
    }
}

let mockup2DGenerator = null

/**
 * @param {HTMLCanvasElement} [renderCanvas]
 * @returns {Mockup2DGenerator}
 */
const getMockup2DGeneratorInstance = () => {
    if (mockup2DGenerator) return mockup2DGenerator

    mockup2DGenerator = new Mockup2DGenerator()
    return mockup2DGenerator
}

const destroyMockup2DGeneratorInstance = () => {
    if (mockup2DGenerator) {
        mockup2DGenerator.destroy()
        mockup2DGenerator = null
    }
}

