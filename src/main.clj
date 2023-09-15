(ns main
  (:require [clojure.math :as math]))

(defn value [v]
  {:val  v
   :grad (atom 0)})

(defn defop [op & params]
  {:val  (apply op (map :val params))
   :kids (set params)
   :op   op
   :grad (atom 0)})

(defn mul-grad [old other]
  (+ old other))

(defn add [a b]
  (defop + a b))

(defn mul [a b]
  (defop * a b))

(defn tanh [x]
  (defop math/tanh x))

(defn init-grad [expr]
  (reset! (expr :grad) 1)
  expr)

(defn update-kid-grad! [expr kid grad]
  (swap! (get-in expr [:kids kid :grad])
         +
         (* @(:grad expr) grad))
  expr)

(defn get-kid-val [expr kid]
  (get-in expr [:kids kid :val]))

(defn backwards [expr]
  (condp = (:op expr)
    nil       expr
    +         (-> expr
                  (update-kid-grad! 0 1)
                  (update-kid-grad! 1 1))
    *         (-> expr
                  (update-kid-grad! 0 (get-kid-val expr 1))
                  (update-kid-grad! 1 (get-kid-val expr 0)))
    math/tanh (-> expr
                  (update-kid-grad! 0 (- 1 (math/pow (:val expr) 2))))))

(defn propagate [expr]
  (clojure.walk/prewalk
    backwards
    (init-grad expr)))

(defn rand-val []
  (dec (rand 2)))

(defn neuron [input-count]
  {:weights (->> rand-val
                 (repeatedly input-count)
                 (map value))
   :bias    (value (rand-val))})

(defn fire [neuron & inputs]
  (->> (:weights neuron)
       (map mul inputs)
       (reduce add (:bias neuron))
       tanh))

(comment

  (clojure.pprint/pprint
    (fire (neuron 3) (value 1) (value 2) (value 3)))

  (def x1 (value 2))
  (def x2 (value 0))
  (def w1 (value -3))
  (def w2 (value 1))
  (def b (value 6.8813735870195432))
  (def x1w1 (mul x1 w1))
  (def x2w2 (mul x2 w2))
  (def x1w1+x2w2 (add x1w1 x2w2))
  (def n (add x1w1+x2w2 b))
  (def o (tanh n))

  (propagate o)

  (def a (value 3))

  (-> (add a a) propagate)

  (-> (add a a) backwards)

  (-> (mul (value 2) (value 3)) propagate)

  (-> 0.8814 value tanh init-grad backwards)

  (propagate (mul (value -1)
                  (add (value 3)
                       (value 2))))

  (clojure.walk/prewalk
    backwards
    (init-grad (mul (value -1)
                    (add (value 3)
                         (value 2)))))

  (backwards (init-grad (mul (value 3)
                             (value 2))))

  (backwards (init-grad (add (value 3)
                             (value 2))))

  (backwards (add (value 3) (value 2)))

  (backwards (value 3))

  (clojure.pprint/pprint
    (mul (value 4)
         (add (value 3)
              (value 2))))

  )
