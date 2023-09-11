(ns main)

(defn value [v]
  {:val  v
   :grad 0})

(defn bin-op [op a b]
  {:val      (op (:val a) (:val b))
   :children [a b]
   :op       op
   :grad     0})

(defn mul-grad [old other]
  (+ old other))

(defn add [a b]
  (bin-op + a b))

(defn mul [a b]
  (bin-op * a b))

(defn backward [expr]
  (condp  = (:op expr)
    nil expr
    +   (-> expr
            (assoc-in [:children 0 :grad] (:grad expr))
            (assoc-in [:children 1 :grad] (:grad expr)))
    *   (-> expr
            (assoc-in [:children 0 :grad] (* (:grad expr) (get-in expr [:children 1 :val])))
            (assoc-in [:children 1 :grad] (* (:grad expr) (get-in expr [:children 0 :val]))))))

(comment

  (backward (assoc (mul (value 3)
                        (value 2))
                   :grad
                   1))

  (backward (assoc (add (value 3)
                     (value 2))
                   :grad
                   1))

  (backward (value 3))

  (->Value 3 [] identity)
  (map->Value {:val 3})

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
