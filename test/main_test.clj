(ns main-test
  (:require [clojure.test :refer [deftest is testing]]
            [main :refer :all]))

(defn- grads [expr]
  (->> (tree-seq :kids :kids expr)
       (map :grad*)
       (map deref)))

(defn- immutate [expr]
  (clojure.walk/postwalk
    (fn [node]
      (if (instance? clojure.lang.Atom node)
        (deref node)
        node))
    expr))

(deftest micrograd-clj-test
  (testing "const"
    (is (= 3 @(:val* (const 3))))
    (is (= 0 @(:grad* (const 3)))))
  (testing "add"
    (is (= nil @(:val* (add (const 2) (const 3)))))
    (is (= 0 @(:grad* (add (const 2) (const 3))))))
  (testing "forward!"
    (is (= 5 @(:val* (forward! (add (const 2) (const 3))))))
    (is (= 20 @(:val* (forward! (mul (const 4) (add (const 2) (const 3))))))))
  (testing "init-grad!"
    (is (= 1 @(:grad* (init-grad! (const 0))))))
  (testing "derive-kids!"
    (is (= [1 1 1] (->> (add (const 2) (const 3))
                        (forward!)
                        (init-grad!)
                        (derive-kids!)
                        (grads)))))
  (testing "backward!"
    (is (= [1 1 1] (->> (add (const 2) (const 3))
                        (forward!)
                        (backward!)
                        (grads))))
    (is (= [1 5 4 4 4] (->> (mul (const 4) (add (const 2) (const 3)))
                            (forward!)
                            (backward!)
                            (grads)))))
  (testing "zero!"
    (is (= [0 0 0] (->> (add (const 2) (const 3))
                        (forward!)
                        (backward!)
                        (zero!)
                        (grads)))))
  (testing "neuron"
    (is (let [n (neuron 2)]
          (and (= [:val* :grad* :val* :grad*] (flatten (map keys (:weights (neuron 2)))))
               (= [:val* :grad*] (keys (:bias n)))))))
  (testing "ready-neuron"
    (is (ready-neuron (neuron 1) [(const 3)])))
  (testing "ready-layer"
    (is (not= nil (ready-layer (layer 1 1) [(const 3)]))))
  (testing "multi-layer-perceptron"
    (is (= 1 (count (multi-layer-perceptron 1 [1])))))
  (testing "ready-perceptron"
    (is (not= nil (ready-perceptron (multi-layer-perceptron 1 [1]) [(const 3)]))))
  (testing "loss"
    (is (= 4.0 @(:val* (forward! (loss [(const 2)] [(const 4)])))))
    (is (not= nil (forward! (loss [(const 1)] (ready-perceptron (multi-layer-perceptron 1 [1]) [(const 3)])))))
    (is (not= nil (forward! (loss [(const 4)] (ready-perceptron (multi-layer-perceptron 2 [2 1]) [(const 3) (const 5)]))))))
  )


(comment

  (def expr (mul (const 4) (add (const 2) (const 3))))

  (get-in expr [:kids 1])

  (clojure.walk/prewalk
    debug
    expr)

  (def a (const 3))

  (def e {:kids [a a]})

  (def e (assoc-in e [:kids 0 :grad*] 1))

  (= (get-in e [:kids 0]) (get-in e [:kids 1]))

  (reduce
    (fn [e v]
      (let [node (get-in e v)
            value (apply (:op node) (map :val* (:kids node)))]
        (assoc-in e (conj v :val*) value)))
    expr
    '([:kids 1] []))

)
