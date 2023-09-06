(ns main)

(defrecord Value [data children op])

(defn add [v1 v2]
  {:data    (+ (:data v1)
               (:data v2))
   :children [v1 v2]
   :op       '+'})

(comment

  (->Value 3 [] identity)
  (map->Value {:data 3})

  (add (Value. 3 [] identity)
       (Value. 2 [] identity))

  (defn f [x]
    (- (* 3 (Math/pow x 2))
      (* 4 x)
      -5))
    (f 3)

  (def xs (range -5 5 1/4))
  (def ys (map f xs))
  )
