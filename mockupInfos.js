const data = {
    'Youth Sweatshirt Gildan 18000B (Made in US)': '18000b',
    'Unisex Jersey Tank Bella Canvas 3480 (Made in US)': '3480',
    'Classic Unisex Crew-neck Sweatshirt Comfort Colors 1566 (Made in US)': '1566',
    'Unisex V-neck T-shirt Gildan 64V00 (Made in EU)': '64v00',
    'Unisex Jersey Short Sleeve Tee Bella Canvas 3001 (Made in US)': '3001',
    'Heavyweight Youth T-Shirt Comfort Colors 9018 (Made in US)': '9018',
    'Classic Unisex T-Shirt Comfort Colors 1717 (Made in US)': '3001',
}

const getAvailableMockups = () => {
    return Object.keys(data)
}

const loadMockupInfos = async (product) => {
    if (data[product]) {
        const res = await fetch(`./data/${data[product]}.json`)
        const json = await res.json()
        return json.mockup_infos
    } else throw new Error('Product not supported')
}
