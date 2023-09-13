(ns main)

(defn value [v]
  {:val  v
   :grad 0})

(defn bin-op [op a b]
  {:val  (op (:val a) (:val b))
   :kids [a b]
   :op   op
   :grad 0})

(defn mul-grad [old other]
  (+ old other))

(defn add [a b]
  (bin-op + a b))

(defn mul [a b]
  (bin-op * a b))

(defn backwards [expr]
  (condp  = (:op expr)
    nil expr
    +   (-> expr
            (assoc-in [:kids 0 :grad] (:grad expr))
            (assoc-in [:kids 1 :grad] (:grad expr)))
    *   (-> expr
            (assoc-in [:kids 0 :grad] (* (:grad expr) (get-in expr [:kids 1 :val])))
            (assoc-in [:kids 1 :grad] (* (:grad expr) (get-in expr [:kids 0 :val]))))))

(defn propagate [expr]
  (clojure.walk/prewalk
    backwards
    (assoc expr :grad 1)))

(comment

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
