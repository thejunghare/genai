console.log('Hello TensorFlow');

async function getData(){
    const response = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const data = await response.json();
    const cleanedData = data.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));
    return cleanedData;
}

async function run(){
    const data = await getData();
    const values = data.map(d =>({
        x : d.horsepower,
        y : d.mpg
    }))

    tfvis.render.scatterplot(
        {name: "Horsepower v MPG"},
        {values},
        {
            xLabel: "Horsepower",
            yLabel : "MPG",
            height : 300
        }
    )
}

document.addEventListener('DOMContentLoaded', run);


// , Menlo, Monaco, 'Courier New', monospace