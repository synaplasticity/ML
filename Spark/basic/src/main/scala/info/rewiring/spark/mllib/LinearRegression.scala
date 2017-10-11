package info.rewiring.spark.mllib

import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

case class LinearRegression(data: RDD[String], numOfIter: Int, learnRate: Double) {

//  def getLabeledPoint(): LabeledPoint = {
//
//    val dataAsString = data.flatMap(line => line.split("\n"))
//                        .map(w => w.split(",")) // Array(Array(1.123, 1))
//
//    // convert values of the array to double
//    val inDouble = dataAsString.map(outer =>
//                                        outer.map(inner => inner.toDouble))
//
//
//    val labelPoint: LabeledPoint = dataAsString foreach (arr =>
//      LabeledPoint(arr(1).toDouble, Vectors.dense(arr(0).toDouble))
//    )
//
//  }


  def getData(): RDD[LabeledPoint] = {

    val parsedData = data.map { line =>
      val x : Array[String] = line.replace("\n", " ").split(" ")
      LabeledPoint.parse(x(0))
    }.cache()

    parsedData
  }
//  def getData(): RDD[LabeledPoint] = {
//    val parsedData = data.map { line =>
//      val x : Array[String] = line.replace("\n", " ").split(" ")
//      val y = x.map{ (a => a.toDouble)}
//      val d = y.size - 1
//      val c = Vectors.dense(y(0),y(d))
//      LabeledPoint(y(0), c)
//    }.cache()
//
//    parsedData
//  }

//  val aofA = x.map(l => l.split(","))
//  val aOfADouble = aofA.map(o => o.map(i => i.toDouble))
//  aOfADouble foreach(a => println(a(0)))
}
