PROGRAM compute_split_matrix

    USE phigemm

    IMPLICIT NONE

    DOUBLE PRECISION, ALLOCATABLE   ::  A(:,:), B(:,:), C(:,:)
    DOUBLE PRECISION                :: alpha, beta
    INTEGER               :: m, n, k, lda, ldb, ldc
    CHARACTER*1           :: transa, transb

    INTEGER     :: ierr, counter

    DOUBLE PRECISION    :: time_start, time_stop
    DOUBLE PRECISION    :: times(1:3)
    DOUBLE PRECISION    :: collects(1:1000,1:2)

    REAL    :: mysplit(1:4)
    REAL    :: x

    INTEGER, EXTERNAL           :: cublas_init, cublas_shutdown
    EXTERNAL                    :: cublas_DGEMM
    DOUBLE PRECISION, EXTERNAL  :: cclock

    m = 2206
    k = 146
    n = 3972
    transa = 'T'
    transb = 'N'

!        int lda = m;
!        int ldb = k;
!        if ( is_transa[ count ] ) lda = k;
!        if ( is_transb[ count ] ) ldb = n;

    lda = k
    ldb = k
    ldc = m

    alpha = 1.0D0
    beta = 1.0D0
    ALLOCATE( A(m,k), B(k,n), C(m,n) )

    A(:,:) = 1.0D0
    B(:,:) = 1.0D0
    C(:,:) = 1.0D0

!    ierr = cublas_init()
    time_start = cclock()
    CALL cublas_DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    time_stop = cclock()
!    ierr = sublas_shutdown()
    times(1) = time_stop-time_start

    time_start = cclock()
    CALL DGEMM (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    time_stop = cclock()
    times(2) = time_stop-time_start

#if defined __PERFORM_PHIGEMM_INIT
    CALL InitCudaEnv()
#endif

    counter = 1
    collects(:,1) = 1000
    collects(:,2) = 0

    DO x = 0.4, 0.975, 0.025

        mysplit(:) = x

#if defined __PHIGEMM_EXPLICIT_SPLITFACTOR
        CALL phigemmsetsplitfactor (mysplit)
#endif

        time_start = cclock()
        CALL phiDgemm (transa,transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        time_stop = cclock()
        collects(counter,1) = time_stop-time_start
        collects(counter,2) = x

        WRITE (*, '("time=",F8.6," split=",F8.6)') collects(counter,1) , collects(counter,2)

        counter = counter + 1
    END DO

#if defined __PERFORM_PHIGEMM_INIT
    CALL CloseCudaEnv()
#endif

    WRITE (*, '(/5x,"CUBLAS=",F8.6," BLAS=",F8.6," PHIGEMM=",F8.6,"(split=",F8.6,")")') times(1), times(2), MINVAL(collects(:,1)) , collects(MINLOC(collects(:,1)),2)

    DEALLOCATE( A, B, C)

END PROGRAM compute_split_matrix
