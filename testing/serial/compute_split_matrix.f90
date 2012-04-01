#if defined __PHIGEMM_PROFILE
#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)
#define phiDgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) phiDgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC,__FILE__,__LINESTR__)
#endif

#define mapping(x) 1000*x

PROGRAM compute_split_matrix

    USE phigemm

    IMPLICIT NONE

    DOUBLE PRECISION, ALLOCATABLE   ::  A(:,:), B(:,:), C(:,:)
    DOUBLE PRECISION                :: alpha, beta
    INTEGER               :: m, n, k, lda, ldb, ldc
    CHARACTER*1           :: transa, transb

    INTEGER     :: ierr, counter
    INTEGER     :: tmp(1)
    INTEGER     :: im, in, ik, index, outer_loop

    DOUBLE PRECISION    :: time_start, time_stop
    DOUBLE PRECISION    :: times(1:3)
    DOUBLE PRECISION    :: collects(1:1000,1:2)

    REAL    :: mysplit(1:4)
    REAL    :: x

    REAL   :: split_matrix(1:10,1:10,1:10)
    REAL   :: best

    INTEGER, EXTERNAL           :: cublas_init, cublas_shutdown
    EXTERNAL                    :: cublas_DGEMM
    DOUBLE PRECISION, EXTERNAL  :: cclock

    alpha = 3.14D0
    beta = -1.1D0

    ! let's compute the best split factor based on the performance

    m = 1025
    k = 1025
    n = 1025
    transa = 'N'
    transb = 'N'
    ! int lda = m;
    ! int ldb = k;
    ! if ( is_transa[ count ] ) lda = k;
    ! if ( is_transb[ count ] ) ldb = n;
    lda = m
    ldb = k
    ldc = m

    ALLOCATE( A(m,k), B(k,n), C(m,n) )

    A(:,:) = 0.9D0
    B(:,:) = 2.0D0
    C(:,:) = 1.3D0

    time_start = cclock()
    CALL cublas_DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    time_stop = cclock()
    times(1) = REAL(time_stop-time_start)

    time_start = cclock()
    CALL DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    time_stop = cclock()
    times(2) = REAL(time_stop-time_start)

    best = (( 2.e-9 ) * m * n * k / times(2)) / ((( 2.e-9 ) * m * n * k / times(2)) + (( 2.e-9 ) *  m * n * k / times(1)) );
!    best = times(1) / (times(1) + times(2));
    DEALLOCATE( A, B, C)

    WRITE (*,*) "The best split factor is = ", best

    DO im=1,3
        DO ik=1,3
            DO in=1,3

			    m = mapping(im)
			    k = mapping(ik)
			    n = mapping(in)
			    transa = 'N'
			    transb = 'N'
			    ! int lda = m;
			    ! int ldb = k;
			    ! if ( is_transa[ count ] ) lda = k;
			    ! if ( is_transb[ count ] ) ldb = n;
			    lda = m
			    ldb = k
			    ldc = m

			    ALLOCATE( A(m,k), B(k,n), C(m,n) )

			    A(:,:) = 1.0D0
			    B(:,:) = 1.0D0
			    C(:,:) = 1.0D0

			    time_start = cclock()
			    CALL cublas_DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
			    time_stop = cclock()
			    times(1) = REAL(INT(REAL(time_stop-time_start) * 1000.0 + 0.5)) / 1000.0

			    time_start = cclock()
			    CALL DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
			    time_stop = cclock()
			    times(2) = REAL(INT(REAL(time_stop-time_start) * 1000.0 + 0.5)) / 1000.0

#if defined __PERFORM_PHIGEMM_INIT
		        CALL InitCudaEnv()
#endif

			    counter = 1
			    collects(:,1) = 1000
			    collects(:,2) = 0

			    DO x = 0.4, 0.975, 0.025

                    A(:,:) = 1.0D0
                    B(:,:) = 1.0D0
                    C(:,:) = 1.0D0

			        mysplit(:) = x

#if defined __PHIGEMM_EXPLICIT_SPLITFACTOR
		          CALL phigemmsetsplitfactor (mysplit)
#endif

			        time_start = cclock()
			        CALL phiDgemm (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
			        time_stop = cclock()
			        collects(counter,1) = REAL(INT(REAL(time_stop-time_start) * 1000.0 + 0.5)) / 1000.0
			        collects(counter,2) = x

			        WRITE (*, '("time=",F8.6," split=",F8.6)') collects(counter,1) , collects(counter,2)

			        counter = counter + 1



			    END DO

#if defined __PERFORM_PHIGEMM_INIT
		        CALL CloseCudaEnv()
#endif

                IF (MINVAL(collects(:,1)) < times(1) .and. MINVAL(collects(:,1)) < times(2)) THEN
                    tmp = MINLOC(collects(:,1))
                    split_matrix(im,in,ik) = REAL(collects(tmp(1),2))
                    WRITE (*, '(5X,"m=",I6," n=",I6," k=",I6," CUBLAS=",F6.4," BLAS=",F6.4," PHIGEMM=",F6.4,"(split=",F6.4,") >> phiGEMM")') m, n, k, times(1), times(2), MINVAL(collects(:,1)) , collects(MINLOC(collects(:,1)),2)
                ENDIF

                IF (times(1) <= MINVAL(collects(:,1)) .and. times(1) < times(2)) THEN
                    split_matrix(im,in,ik) = -1.0
                    WRITE (*, '(5X,"m=",I6," n=",I6," k=",I6," CUBLAS=",F6.4," BLAS=",F6.4," PHIGEMM=",F6.4,"(split=",F6.4,") >> CUBLAS")') m, n, k, times(1), times(2), MINVAL(collects(:,1)) , collects(MINLOC(collects(:,1)),2)
                ENDIF

                IF (times(2) <= MINVAL(collects(:,1)) .and. times(2) <= times(1)) THEN
                     split_matrix(im,in,ik) = 0.0
                    WRITE (*, '(5X,"m=",I6," n=",I6," k=",I6," CUBLAS=",F6.4," BLAS=",F6.4," PHIGEMM=",F6.4,"(split=",F6.4,") >> BLAS")') m, n, k, times(1), times(2), MINVAL(collects(:,1)) , collects(MINLOC(collects(:,1)),2)
                ENDIF

                DEALLOCATE( A, B, C)
		    ENDDO
	    ENDDO
    ENDDO

    ! pathetic loop to print information...
!    DO outer_loop=1,10
!        WRITE(*,*) "m=", mapping(outer_loop)
!        DO index=1,10
!            WRITE(*,*) "n=", mapping(index)
!            WRITE(*,*) split_matrix(outer_loop,index,:)
!        ENDDO
!    ENDDO

END PROGRAM compute_split_matrix
