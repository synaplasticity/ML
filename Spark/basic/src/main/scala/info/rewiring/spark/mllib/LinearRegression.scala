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
      val x : Array[String] = line.replace("\n", " ").split(" ")
      LabeledPoint.parse(x(0))
    }.cache()

    parsedData
  }

  def train(trainingData: RDD[LabeledPoint]): LinearRegressionModel = {
//    LinearRegressionWithSGD.train(trainingData, numOfIter, learnRate)
    LinearRegressionWithSGD.train(trainingData, numOfIter, learnRate, 0.35, Vectors.dense(0.0))
  }



//  val aofA = x.map(l => l.split(","))
//  val aOfADouble = aofA.map(o => o.map(i => i.toDouble))
//  aOfADouble foreach(a => println(a(0)))
}
