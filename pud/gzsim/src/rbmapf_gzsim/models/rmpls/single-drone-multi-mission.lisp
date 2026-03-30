(defpackage #:scenario1)
(in-package #:scenario1)

(define-control-program start-mission-one-drone-one ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 120)
	:contingent t
)))

(define-control-program start-mission-two-drone-one ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 120)
	:contingent t
)))

(define-control-program sync ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 60)
	:contingent t
)))

(define-control-program land-drone-one ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 5)
)))

(define-control-program znoop-one ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 1)
)))

(define-control-program znoop-two ()
	(declare (primitive)
	(duration (simple :lower-bound 0 :upper-bound 1)
)))

(define-control-program main ()
	(with-temporal-constraint (simple-temporal :upper-bound 500)
	(sequence (:slack nil)
		(start-mission-one-drone-one)
		(znoop-one)
		(sync)
		(znoop-two)
		(start-mission-two-drone-one)
		(land-drone-one)
	))
)
