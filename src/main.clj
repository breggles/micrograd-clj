(ns main
  (:require [clojure.math :as math]))

(defn const [v]
  {:val  (atom v)
   :grad (atom 0)})

(defn- defop [op & params]
  {:val  (atom nil)
   :kids (vec params)
   :op   op
   :grad (atom 0)})

(defn add [a b]
  (defop + a b))

(defn mul [a b]
  (defop * a b))

(defn neg [a]
  (mul a (const -1)))

(defn sub [a b]
  (add a (neg b)))

(defn pow [a b]
  (defop math/pow a b))

(defn tanh [x]
  (defop math/tanh x))

(defn forward! [expr]
  (->> (tree-seq :kids :kids expr)
       (distinct)
       (filter :kids)
       (reverse)
       (map
         (fn [node]
           (reset!
             (:val node)
             (apply
               (:op node)
               (map (comp deref :val) (:kids node))))))
       (dorun))
  expr)

(defn init-grad! [expr]
  (reset! (expr :grad) 1)
  expr)

(defn update-kid-grad! [expr kid grad]
  (swap! (get-in expr [:kids kid :grad])
         +
         (* @(:grad expr) grad))
  expr)

(defn get-kid-val [expr kid]
  @(get-in expr [:kids kid :val]))

(defn debug [x] (clojure.pprint/pprint x) x)

(defn derive-kids! [expr]
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
                  (update-kid-grad! 0 (- 1 (math/pow @(:val expr) 2))))))

(defn backward! [expr]
  (->> (init-grad! expr)
       (tree-seq :kids :kids)
       (distinct)
       (map derive-kids!)
       (dorun)
       )
  expr)

(defn zero! [expr]
  (->> (tree-seq :kids :kids expr)
       (distinct)
       (map (comp #(reset! % 0) :grad))
       (dorun))
  expr)

(defn- rand-val []
  (dec (rand 2)))

(defn neuron [input-count]
  {:weights (->> rand-val
                 (repeatedly input-count)
                 (map const))
   :bias    (const (rand-val))})

(defn fire-neuron [neuron inputs]
  (->> (:weights neuron)
       (map mul inputs)
       (reduce add (:bias neuron))
       tanh))

(defn neuron-params [neuron]
  (conj (:weights neuron) (:bias neuron)))

(defn layer [input-count neuron-count]
  (repeatedly neuron-count #(neuron input-count)))

(defn fire-layer [layer inputs]
  (map #(fire-neuron % inputs) layer))

(defn layer-params [layer]
  (flatten (map neuron-params layer)))

(defn multi-layer-perceptron [input-count neuron-counts]
  (->> (cons input-count neuron-counts)
       (partition 2 1)
       (map (partial apply layer))))

(defn fire-perceptron [perceptron inputs]
  (reduce #(fire-layer %2 %1) inputs perceptron))

(defn perceptron-params [perceptron]
  (set (flatten (map layer-params perceptron))))

(defn predict [perceptron inputs]
  (map first
    (map (partial fire-perceptron perceptron) inputs)))

(defn loss [targets predictions]
  (->> (map sub predictions targets)
       (map #(pow % (const 2)))
       (reduce add)))

(defn update-params [expr params]
  (clojure.walk/postwalk
    (fn [inner]
      (if (params inner)
        (update inner :val
          (fn [old-val *grad] (+ old-val (* -0.01 @*grad)))
          (:grad inner))
        inner))
    expr))

(comment

  (def inputs (map (partial map (partial const))
              [[2    3 -1  ]
               [3   -1  0.5]
               [0.5  1  1  ]
               [1    1 -1  ]]))

  (def targets (map (partial const) [1 -1 -1 1]))

  (def mlp (multi-layer-perceptron 3 [4 4 1]))

  (def predictions (predict mlp inputs))

  (def l (loss targets predictions))

  (:val l) (:grad l) (backward! (forward! l))

  ((comp first :weights first first) mlp)

  (update-params l (perceptron-params mlp))


  (count (perceptron-params (multi-layer-perceptron 3 [4 4 1])))

  (layer-params (layer 4 3))

  (neuron-params (neuron 3))

  (def t (init-grad! (const 3)))

  (zero! t)

  (zero! (init-grad! (const 3)))

  (clojure.pprint/pprint
    (fire-perceptron
      (multi-layer-perceptron 1 [1])
      [(const 2)]))

  (clojure.pprint/pprint
    (backward!
      (first
        (fire-perceptron
          (multi-layer-perceptron 3 [4 4 1])
          [(const 2) (const 3) (const -1)]))))

  (fire-layer (layer 3 4) [(const 1) (const 2) (const 3)])

  (def x1 (const 2))
  (def x2 (const 0))
  (def w1 (const -3))
  (def w2 (const 1))
  (def b (const 6.8813735870195432))
  (def x1w1 (mul x1 w1))
  (def x2w2 (mul x2 w2))
  (def x1w1+x2w2 (add x1w1 x2w2))
  (def n (add x1w1+x2w2 b))
  (def o (tanh n))

  (get-in o [:kids 0 :grad])

  (tree-seq :kids :kids (mul (const 4) (add (const 2) (const 3))))

  (clojure.pprint/pprint
    (fire-neuron (neuron 3) [(const 1) (const 2) (const 3)]))

  (clojure.pprint/pprint (backward! o))

  (def a (const 3))

  (-> (add a a) backward!)

  (def b a)

  (= b a)

  (reset! (:grad a) 1)

  (-> (mul (const 2) (const 3)) backward!)

  (-> (pow (const 2) (const 3))
      (init-grad!)
      (derive-kids!))

  (-> 0.8814 const tanh init-grad! derive-kids!)

  (backward! (mul (const -1)
                   (add (const 3)
                        (const 2))))

  (clojure.walk/prewalk
    derive-kids!
    (init-grad! (mul (const -1)
                    (add (const 3)
                         (const 2)))))

  (derive-kids! (init-grad! (mul (const 3)
                             (const 2))))

  (derive-kids! (init-grad! (add (const 3)
                             (const 2))))

  (derive-kids! (add (const 3) (const 2)))

  (derive-kids! (const 3))

  (clojure.pprint/pprint
    (mul (const 4)
         (add (const 3)
              (const 2))))

  )
