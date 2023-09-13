(ns main
  (:require [clojure.math :as math]))

(defn value [v]
  {:val  v
   :grad 0})

(defn defop [op & params]
  {:val  (apply op (map :val params))
   :kids (vec params)
   :op   op
   :grad 0})

(defn mul-grad [old other]
  (+ old other))

(defn add [a b]
  (defop + a b))

(defn mul [a b]
  (defop * a b))

(defn tanh [x]
  (defop math/tanh x))

(defn backwards [expr]
  (condp = (:op expr)
    nil expr
    +   (-> expr
            (assoc-in [:kids 0 :grad] (* (:grad expr) 1))
            (assoc-in [:kids 1 :grad] (* (:grad expr) 1)))
    *   (-> expr
            (assoc-in [:kids 0 :grad] (* (:grad expr) (get-in expr [:kids 1 :val])))
            (assoc-in [:kids 1 :grad] (* (:grad expr) (get-in expr [:kids 0 :val]))))
    math/tanh (-> expr
                  (assoc-in [:kids 0 :grad] (* (:grad expr) (- 1 (math/pow (math/tanh (get-in expr [:kids 0 :val])) 2)))))))

(defn propagate [expr]
  (clojure.walk/prewalk
    backwards
    (assoc expr :grad 1)))

(comment

  (backwards
    (assoc (tanh (value 0.8814))
           :grad
           1))

  (propagate (mul (value -1)
                  (add (value 3)
                       (value 2))))

  (clojure.walk/prewalk
    backward
    (assoc (mul (value -1)
                (add (value 3)
                     (value 2)))
           :grad
           1))

  (backward (assoc (mul (value 3)
                        (value 2))
                   :grad
                   1))

  (backward (assoc (add (value 3)
                     (value 2))
                   :grad
                   1))

  (backward (value 3))

  (clojure.pprint/pprint
    (mul (value 4)
         (add (value 3)
              (value 2))))

  (defn f [x]
    (- (* 3 (Math/pow x 2))
      (* 4 x)
      -5))
    (f 3)

  (def xs (range -5 5 1/4))
  (def ys (map f xs))

  )
