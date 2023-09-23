(ns main-test
  (:require [clojure.test :refer [deftest is testing]]
            [main :refer :all]))

(deftest micrograd-clj-test
  (testing "const"
    (is (= 3 @(:val (const 3))))
    (is (= 0 @(:grad (const 3)))))
  (testing "add"
    (is (= nil @(:val (add (const 2) (const 3)))))
    (is (= 0 @(:grad (add (const 2) (const 3))))))
  (testing "forward!"
    (is (= 5 @(:val (forward! (add (const 2) (const 3))))))
    (is (= 20 @(:val (forward! (mul (const 4) (add (const 2) (const 3))))))))
  (testing "init-grad!"
    (is (= 1 @(:grad (init-grad! (const 0))))))
  (testing "derive-kids!"
    (is (= [1 1] (->> (add (const 2) (const 3))
                      (forward!)
                      (init-grad!)
                      (derive-kids!)
                      (:kids)
                      (map (comp deref :grad))))))
  (testing "backward!"
    (is (= [1 1 1] (->> (add (const 2) (const 3))
                        (forward!)
                        (backward!)
                        (grads))))
    (is (= [1 5 4 4 4] (->> (mul (const 4) (add (const 2) (const 3)))
                            (forward!)
                            (backward!)
                            (grads)))))
  )

(defn- grads [expr]
  (->> (tree-seq :kids :kids expr)
       (map :grad)
       (map deref)))

(comment

  (def expr (mul (const 4) (add (const 2) (const 3))))

  (get-in expr [:kids 1])

  (clojure.walk/prewalk
    debug
    expr)

  (def a (const 3))

  (def e {:kids [a a]})

  (def e (assoc-in e [:kids 0 :grad] 1))

  (= (get-in e [:kids 0]) (get-in e [:kids 1]))

  (reduce
    (fn [e v]
      (let [node (get-in e v)
            value (apply (:op node) (map :val (:kids node)))]
        (assoc-in e (conj v :val) value)))
    expr
    '([:kids 1] []))

)
