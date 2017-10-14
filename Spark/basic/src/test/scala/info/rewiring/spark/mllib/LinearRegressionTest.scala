package info.rewiring.spark.mllib

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSpec

class LinearRegressionTest extends FunSpec{
  val sparkConf = new SparkConf()
    .setAppName("Linear regression")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)
  val data = sc.textFile("file:///opt/spark/data/mllib/popvsrev/ex1data1.csv")

  val numOfIters = 75000
  val learnRate = 0.09

  describe("Simple linear regression") {

    describe("Basic load, univariate train and predict tests") {
      it("Should load the provided resources correctly") {
        val expectedResults = Array("17.592,6.1101")
        assert(data.take(1).sameElements(expectedResults))
      }

      it("should create the correct LabeledPoint for the given data") {
        val linearRegression = LinearRegression(data, numOfIters, learnRate)

        val labeledPoint = linearRegression.getData()

        assert(labeledPoint.count() === 97)
        assert(labeledPoint.first().label === 17.592 )
        assert(labeledPoint.first().features === Vectors.dense(6.1101, 1))
      }

      it("should correctly train the model") {
        val linearRegression = LinearRegression(data, numOfIters, learnRate)
        val labeledPoint: RDD[LabeledPoint] = linearRegression.getData()

        val model: LinearRegressionModel = linearRegression.train(labeledPoint)


        println("Weights ------> " + model.weights)
        linearRegression.evaluate(labeledPoint, model)
        assert(model.predict(Vectors.dense(1, 20.0)) === 17.14723903437474)
        assert(model.predict(Vectors.dense(1, 40.0)) === 34.29447806874948)
        assert(model.predict(Vectors.dense(1, 50.0)) === 42.868097585936844)

      }

    }

  }

  override def finalize(): Unit = {
    sc.stop()
  }


}
