(ns main)

(defn value [data]
  {:data data})

(defn bin-op [v1 v2 op]
  {:data    (op (:data v1)
                (:data v2))
   :children [v1 v2]
   :op       'op})

(defn add [v1 v2]
  (bin-op v1 v2 +))

(defn mul [v1 v2]
  {:data    (* (:data v1)
               (:data v2))
   :children [v1 v2]
   :op       *})

(comment

  (->Value 3 [] identity)
  (map->Value {:data 3})

  (mul (value 4)
       (add (value 3)
            (value 2)))

  (defn f [x]
    (- (* 3 (Math/pow x 2))
      (* 4 x)
      -5))
    (f 3)

  (def xs (range -5 5 1/4))
  (def ys (map f xs))
  )
