
(defpackage #:scenario1)
(in-package #:scenario1)
    
(define-control-program start-mission-one-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600)
    :contingent t
)))
            
(define-control-program start-mission-one-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600)
    :contingent t
)))
            
(define-control-program sync-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600)
    :contingent t
)))
            
(define-control-program start-mission-two-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600)
    :contingent t
)))
            
(define-control-program start-mission-two-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600)
    :contingent t
)))
            
(define-control-program land-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))
        
(define-control-program land-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))
        
(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound 2400)
    (sequence (:slack nil)
    
(parallel (:slack t)
            
(start-mission-one-drone-one)
        
(start-mission-one-drone-two)
            )

(sync-one)
            

(parallel (:slack t)
            
(start-mission-two-drone-one)
        
(start-mission-two-drone-two)
            )
(parallel (:slack t)
(land-drone-one)
    
(land-drone-two)
        )
))
)
    