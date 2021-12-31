require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
/**
 * PROBLEM:
 * Given the horsepower, weight, displacement of a vehicle,
 * will it have HIGH, MEDIUM or LOW fuel efficiency?
 *
 * Mile Per Gallon aslında fuel efficiency olarak tabir edilebilir.
 * 0 - 15   MPG -> LOW
 * 15 - 30  MPG -> MEDIUM
 * 30+      MPG -> HIGH
 */

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    converters: {
      mpg: (value) => {
        const mpg = parseFloat(value);
        if (mpg < 15) {
          return [1, 0, 0];
        } else if (mpg < 30) {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      },
    },
  }
);

// Yukarıdaki converter [ [ [ 1, 0, 0 ] ], [ [ 0, 0, 1 ] ] ] gibi sonuç veriyor.
// İçerdeki bir arrayden kurtulmak için lodash flatMap fonksiyonunu kullanıyoruz.
// Sonuç [ [ 0, 1, 0 ], [ 0, 1, 0 ] ]
// console.log(_.flatMap(labels));

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

// regression.weights.print(); // (4 x 3) - 3 Label, [b, m1, m2, m3] - 4 mb

regression.train();
regression.predict([[150, 200, 2.223]]).print();
// console.log(regression.test(testFeatures, testLabels));

console.log(regression.test(testFeatures, _.flatMap(testLabels)));
// plot({
//   x: regression.costHistory.reverse(),
// });
