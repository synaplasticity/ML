package info.rewiring.spark.mllib

import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

case class LinearRegression(data: RDD[String], numOfIter: Int, learnRate: Double) {


  def getData(): RDD[LabeledPoint] = {

    val parsedData = data.map { line =>
      val rows: Array[String] = line.replace("\n", " ").split(" ")
      val x: Array[String] = rows(0).split(",")
      LabeledPoint.parse(x(0) + ",1 " + x(1))
    }.cache()


    parsedData
  }

  def train(trainingData: RDD[LabeledPoint]): LinearRegressionModel = {
    //    LinearRegressionWithSGD.train(trainingData, numOfIter, learnRate)
    LinearRegressionWithSGD.train(trainingData, numOfIter, learnRate, 1.0, Vectors.dense(0.0, 0.0))
  }

  def evaluate(parsedData: RDD[LabeledPoint], model: LinearRegressionModel) = {
    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val MSE = valuesAndPreds.map { case (value, prediction) =>
      math.pow((value - prediction), 2)
    }
      .mean()
    println("training Mean Squared Error = " + MSE)
  }

}
