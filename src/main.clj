(ns main
  (:require [clojure.math :as math]))

(defn value [v]
  {:val  v
   :grad (atom 0)})

(defn defop [op & params]
  {:val  (apply op (map :val params))
   :kids (vec params)
   :op   op
   :grad (atom 0)})

(defn mul-grad [old other]
  (+ old other))

(defn add [a b]
  (defop + a b))

(defn mul [a b]
  (defop * a b))

(defn neg [a]
  (mul a (value -1)))

(defn sub [a b]
  (add a (neg b)))

(defn pow [a b]
  (defop math/pow a b))

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

(defn debug [x] (print x) x)

(defn backwards! [expr]
  (condp = (:op expr)
    nil       expr
    +         (-> expr
                  (update-kid-grad! 0 1)
                  (update-kid-grad! 1 1))
    *         (-> expr
                  (update-kid-grad! 0 (get-kid-val expr 1))
                  (update-kid-grad! 1 (get-kid-val expr 0)))
    math/pow  (-> expr
                  (update-kid-grad! 0 (math/pow (* (get-kid-val expr 1)
                                                   (get-kid-val expr 0))
                                                (dec (get-kid-val expr 1)))))
    math/tanh (-> expr
                  (update-kid-grad! 0 (- 1 (math/pow (:val expr) 2))))))

(defn propagate! [expr]
  (->> (init-grad expr)
       (tree-seq :kids :kids)
       (distinct)
       (map backwards!)
       (dorun))
  expr)

(defn zero! [expr]
  (->> (tree-seq :kids :kids expr)
       (distinct)
       (map (comp #(reset! % 0) :grad))
       (dorun))
  expr)

(defn rand-val []
  (dec (rand 2)))

(defn neuron [input-count]
  {:weights (->> rand-val
                 (repeatedly input-count)
                 (map value))
   :bias    (value (rand-val))})

(defn fire-neuron [neuron inputs]
  (->> (:weights neuron)
       (map mul inputs)
       (reduce add (:bias neuron))
       tanh))

(defn layer [input-count neuron-count]
  (repeatedly neuron-count #(neuron input-count)))

(defn fire-layer [layer inputs]
  (map #(fire-neuron % inputs) layer))

(defn multi-layer-perceptron [input-count neuron-counts]
  (->> (cons input-count neuron-counts)
       (partition 2 1)
       (map (partial apply layer))))

(defn fire-perceptron [perceptron inputs]
  (reduce #(fire-layer %2 %1) inputs perceptron))

(defn loss [targets predictions]
  (reduce add (map (comp #(pow % (value 2)) sub) predictions targets)))


(comment

  (def inputs (map (partial map (partial value))
              [[2    3 -1  ]
               [3   -1  0.5]
               [0.5  1  1  ]
               [1    1 -1  ]]))

  (def targets (map (partial value) [1 -1 -1 1]))

  (def mlp (multi-layer-perceptron 3 [4 4 1]))

  (def predictions (map (comp first (partial fire-perceptron mlp)) inputs))

  (def l (loss targets predictions))

  (:val l) (:grad l) (propagate! l)

  ((comp first :weights first first) mlp)


  (def t (init-grad (value 3)))

  (zero! t)

  (zero! (init-grad (value 3)))

  (clojure.pprint/pprint
    (fire-perceptron
      (multi-layer-perceptron 1 [1])
      [(value 2)]))

  (clojure.pprint/pprint
    (propagate!
      (first
        (fire-perceptron
          (multi-layer-perceptron 3 [4 4 1])
          [(value 2) (value 3) (value -1)]))))

  (fire-layer (layer 3 4) [(value 1) (value 2) (value 3)])

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

  (get-in o [:kids 0 :grad])

  (tree-seq :kids :kids (mul (value 4) (add (value 2) (value 3))))

  (clojure.pprint/pprint
    (fire-neuron (neuron 3) [(value 1) (value 2) (value 3)]))

  (clojure.pprint/pprint (propagate! o))

  (def a (value 3))

  (-> (add a a) propagate!)

  (def b a)

  (= b a)

  (reset! (:grad a) 1)

  (-> (mul (value 2) (value 3)) propagate!)

  (-> (pow (value 2) (value 3))
      (init-grad)
      (backwards!))

  (-> 0.8814 value tanh init-grad backwards!)

  (propagate! (mul (value -1)
                   (add (value 3)
                        (value 2))))

  (clojure.walk/prewalk
    backwards!
    (init-grad (mul (value -1)
                    (add (value 3)
                         (value 2)))))

  (backwards! (init-grad (mul (value 3)
                             (value 2))))

  (backwards! (init-grad (add (value 3)
                             (value 2))))

  (backwards! (add (value 3) (value 2)))

  (backwards! (value 3))

  (clojure.pprint/pprint
    (mul (value 4)
         (add (value 3)
              (value 2))))

  )
