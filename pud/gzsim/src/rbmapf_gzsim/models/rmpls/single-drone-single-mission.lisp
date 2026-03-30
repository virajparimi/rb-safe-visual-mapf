(defpackage #:scenario1)
(in-package #:scenario1)

(define-control-program start-mission-drone-one ()
	(declare (primitive)
	(duration (simple :lower-bound 5 :upper-bound 10)
	:contingent t
)))
	
(define-control-program land-drone-one ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 5)
)))

(define-control-program main ()
	(with-temporal-constraint (simple-temporal :upper-bound 15)
	(sequence (:slack nil)
		(start-mission-drone-one)
		(land-drone-one)
	))
)
