name := "basic"

version := "0.1"

//scalaVersion := "2.12.3"
scalaVersion := "2.11.8"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"

//libraryDependencies ++= Seq(
//  "org.apache.spark" %% "spark-core" % "2.2.0"
//)



resolvers += "Artima Maven Repository" at "http://repo.artima.com/releases"

//resolvers ++= Seq(
//  "Artima Maven Repository" at "http://repo.artima.com/releases",
//  Resolver sonatypeRepo "public",
//  Resolver typesafeRepo "releases"
//)

