class loss:
    """
        1-Forward function 
        error of X with respect to Y_labels.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
            Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
        Returns:
            loss: numpy.float. Mean square error of x.


        2- Local gradient function
        Local gradient with respect to X at (X, Y).
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim) which (WX).
            Y: numpy.ndarray of shape (n_batch, n_dim) which (Y_labels).
        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim) which delta .
        """
pass 