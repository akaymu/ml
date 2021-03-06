require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
  }
);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();
// const r2 = regression.test(testFeatures, testLabels);
// console.log(`R^2: ${r2}`);

// plot({
//   x: regression.bHistory,
//   y: regression.mseHistory.reverse(),
//   xLabel: 'Value of B',
//   yLabel: 'Mean Squared Error',
// });
plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
});

regression
  .predict([
    [120, 2, 380],
    [135, 2.1, 420],
  ])
  .print();
