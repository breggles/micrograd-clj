(ns micrograd-test
  (:require [clojure.test :refer [deftest is testing]]
            [micrograd :refer :all]))

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
  (testing "div"
    (is (= 2.0 @(:val* (forward! (div (const 6) (const 3)))))))
  (testing "log"
    (is (= 0.6931471805599453 @(:val* (forward! (log (const 2))))))
    (is (= 1.4426950408889634 @(get-in (backward! (forward! (log (const 2))))
                                       [:kids 0 :grad*]))))
  (testing "tanh"
    (is (= nil @(:val* (tanh (const 2)))))
    (is (= 0 @(:grad* (tanh (const 2))))))
  (testing "forward!"
    (is (= 5 @(:val* (forward! (add (const 2) (const 3))))))
    (is (= 20 @(:val* (forward! (mul (const 4) (add (const 2) (const 3)))))))
    (is (clojure.string/starts-with? (str @(:val* (forward! (tanh (const 2)))))
                                     "0.96402")))
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
    (is (= [1 1] (->> (add (const 2))
                      (forward!)
                      (backward!)
                      (grads))))
    (is (= [1 7.38905609893065] (->> (exp (const 2))
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
    (is (not= nil
              ((juxt :val* :grad*) (let [mlp (multi-layer-perceptron 2 [2 2 1])]
                (forward!
                  (loss [(const 4) (const 5)]
                        [(first (ready-perceptron mlp [(const 3) (const 5)]))
                         (first (ready-perceptron mlp [(const 1) (const -1)]))])))))))
  (testing "one-hot"
    (is (= [0 1 0] (mapv (comp deref :val*) (one-hot 3 1))))
    (is (= [1 0 0] (mapv (comp deref :val*) (one-hot 3 0))))
  (testing "normalize-row"
    (is (= [0.16666666666666666
            0.3333333333333333
            0.5]
           (->> (normalize-row [(const 1) (const 2) (const 3)])
                (mapv forward!)
                (mapv (comp deref :val*)))))
    (is (= 1.0
           (->> (normalize-row [(const 1) (const 2) (const 3)])
                (mapv forward!)
                (mapv (comp deref :val*))
                (apply +)))))
  (testing "mean"
    (is (= 3.0 @(:val* (forward! (mean [(const 2) (const 4)]))))))
  ))


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
