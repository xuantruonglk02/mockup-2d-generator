const artworkList = [
    "transparent.png",
    "artwork_1.jpeg",
    "white.png"
];

const getAvailableArtworks = () => {
    return artworkList.map(name => ({
        name: name,
        url: `./artworks/${name}`
    }));
};
