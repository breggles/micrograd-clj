(ns main-test
  (:require [clojure.test :refer [deftest is testing]]))

(deftest micrograd-clj-test
  (testing "The micrograds"
    (is (= 1 2))))
