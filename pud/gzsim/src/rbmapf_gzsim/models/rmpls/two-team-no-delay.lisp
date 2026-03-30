(defpackage #:scenario1)
(in-package #:scenario1)
    
(define-control-program start-mission-one-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 5
	:max-observation-delay 10
    )
    :contingent t
)))
            
(define-control-program start-mission-one-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 5
	:max-observation-delay 10
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 5
	:max-observation-delay 10
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 5
	:max-observation-delay 10
    )
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

(define-control-program land-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program land-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program upload-a ()
    (declare (primitive)
    (duration (simple :lower-bound 2 :upper-bound 3)
)))

(define-control-program upload-b ()
    (declare (primitive)
    (duration (simple :lower-bound 2 :upper-bound 3)
)))

(define-control-program sync-one ()
    (declare (primitive)
    (duration (simple :lower-bound 0 :upper-bound 0)
)))

(define-control-program sync-two ()
    (declare (primitive)
    (duration (simple :lower-bound 0 :upper-bound 0)
)))
 
        
(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound 2400)
    (sequence (:slack nil)
        
        (parallel (:slack t)
            (start-mission-one-drone-one)
            (start-mission-one-drone-two)
        )

        (upload-a)

        (parallel (:slack t)
            (land-drone-one)
            (land-drone-two)
        )


	    (sync-one)

        (parallel (:slack t)
            (start-mission-two-drone-three)
            (start-mission-two-drone-four)
        )

        (upload-b)

        (parallel (:slack t)
            (land-drone-three)
            (land-drone-four)
        )
        
    ))
)
