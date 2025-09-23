const data = {
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
