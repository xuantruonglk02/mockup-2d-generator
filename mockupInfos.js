const data = {
    // part 2
    'Youth Jersey Tee Bella Canvas 3001Y (Made in US)': '3001yus',
    'Classic Unisex T-Shirt Gildan 5000 (Made in AU)': '5000au',
    'Youth T-shirt Gildan 64000B (Made in US)': '64000bus',
    'Comfort Color Garment-Dyed Lightweight Fleece Crewneck Sweatshirt 1466 (Made in US)': '1466us',
    'Classic Unisex Zip Hoodie Gildan 18600 (Made in US)': '18600us',
    'Classic Unisex  Performance T-Shirt Gildan 42000 (Made in US)': '42000us',
    'Zip Hoodie Gildan 18600 (Made in AU)': '18600au',
    'Heavy Cotton Tank Top Gildan 5200  (Made in US)': '5200us',
    'Women’s Relaxed Jersey V-Neck Tee Bella Canvas 6405 (Made in US)': '6405us',
    'Women\'s Ideal T-Shirt Next Level 1510 (Made in US)': '1510us',
    
    // part 1
    'polo': 'polo',
    'Youth Sweatshirt Gildan 18000B (Made in US)': '18000bus',
    'Classic Unisex Crew-neck Sweatshirt Comfort Colors 1566 (Made in US)': '1566us',
    'Women\'s T-shirt Gildan 5000L (Made In US)': '5000lus',
    'Unisex Jersey Tank Bella Canvas 3480 (Made in US)': '3480us',
    'Heavyweight Youth T-Shirt Comfort Colors 9018 (Made in US)': '9018us',
    'Unisex Jersey Short Sleeve Tee Bella Canvas 3001 (Made in US)': '3001us',
    'Unisex V-neck T-shirt Gildan 64V00 (Made in EU)': '64v00eu',
    'Unisex V-neck T-shirt Bella Canvas 3005 (Made in US)': '3005us',
    'Youth T-shirt Gildan 5000B (Made in EU)': '5000beu',
    'Youth T-shirt Gildan 5000B (Made in AU)': '5000bau',
    'Classic Unisex Hoodie Comfort Colors 1567 (Made in US)': '1567us',
    'Classic Unisex T-Shirt Gildan 5000 (Made In US)': '5000us',
    'Baby Bodysuit LAT 4424 (Made in US)': '4424us',
    'Classic Unisex T-Shirt Gildan 6400 (Made In AU)': '6400au',
    'Classic Unisex Hoodie Gildan 18500 (Made In AU)': '18500au',
    'Long Sleeve T-shirt Gildan 2400 (Made in AU)': '2400au',
    'Women\'s T-shirt': 'wts',
    'Youth T-shirt': 'yts',
    'Women\'s V-neck T-shirt': 'wvts',
    'Youth Sweatshirt AWDIS JH30J (Made in EU)': 'jh30jeu',
    'Zip Hoodie Gildan 18600 (Made in EU)': '18600eu',
    'Ladies\' V-neck T-shirt Gildan 5V00L (Made in US)': '5v00lus',
    'Long Sleeve T-Shirt Gildan 5400 (Made in US)': '5400us',
    'Ladies’ Tank Top Gildan 64200L (Made in EU)': '64200leu',
    'Bleach Splatter Unisex V-neck T-shirt': 'bsvts',
    'Kid Hoodie AWDis JH001J (Made in EU)': 'jh001jeu',
    'Heavyweight Adult Pocket T-Shirt Comfort Colors 6030 (Made in US)': '6030us',
    'Toddler Fine Jersey Rabbit Skins 3321 (Made in US)': '3321us',
    'Premium Unisex Hoodie AS Colour 5101 (Made In AU)': '5101au',
    'Classic Unisex DryBlend T-Shirt Gildan 8000 (Made in US)': '8000us',
}

const getAvailableMockups = () => {
    return Object.keys(data)
}

const loadMockupInfos = async (product, useOptimized) => {
    if (data[product]) {
        const res = useOptimized
            ? await fetch(`./optimized-mockup-infos/${data[product]}/mockup_infos.optimized_web.json`)
            : await fetch(`./data/${data[product]}.json`)
        const json = await res.json()
        return json.mockup_infos
    } else throw new Error('Product not supported')
}
