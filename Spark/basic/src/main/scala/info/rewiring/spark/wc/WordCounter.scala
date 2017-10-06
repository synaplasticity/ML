package info.rewiring.spark.wc

import org.apache.spark.rdd.RDD

case class Words(doc: RDD[String]) {


//  def count: RDD[(Int, String)] = {
  def count: RDD[(Int, String)] = {


    // use the filter function to remove lines with no text
    val flattened: RDD[String] = getFlattened(doc)

    // Map text to lowercase, remove empty strings, and then convert to
    // key value pairs in the form of (word, 1)
    val kvpairs: RDD[(String, Int)] = getPairs(flattened)

    val countByWords = getCountedReveresedOrder(kvpairs)

    countByWords.map(t => t.swap).sortByKey(false)

  }

  def getPairs(flattened: RDD[String]): RDD[(String, Int)] = {
    // Map text to lowercase, remove empty strings, and then convert to
    // key value pairs in the form of (word, 1)
    flattened.filter(w => w.length > 0)
      .map(w => (w.toLowerCase(), 1))
  }

  def getFlattened(doc: RDD[String]) = {
    // use the filter function to remove lines with no text
    doc.filter(l => l.length > 0)
      .flatMap(l => l.split("\\W+"))
  }

  def getCountedReveresedOrder(kvpairs: RDD[(String, Int)]): RDD[(String, Int)] = {
    // Count each word and sort results in reverse alphabetic order:
    kvpairs.reduceByKey((total, next) => total+next).sortByKey(false)
  }


}
