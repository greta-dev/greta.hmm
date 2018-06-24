#' @name simulate_hmm
#' @title Simulate Sequences of Hidden and Observed States from a Hidden Markov
#'   Model
#'
#' @description Generate corresponding sequences of hidden and observed states
#'   from a Hidden Markov Model (HMM) with specified transition and emission
#'   matrices. The simulation is carried out directly in R and is provided only
#'   to generate datasets for demonstrating or testing HMMs.
#'
#' @export
#'
#' @param transition a K x K matrix of transition probabilities between hidden
#'   states
#' @param emission a K x N matrix of emission probabilities between hidden and
#'   observed states
#' @param n_timesteps the number of timesteps (length of the observed state
#'   matrix) - must be a positive scalar integer
#'
#' @examples
#'
#' # numbers of hidden and observable states
#' n_hidden <- 3
#' n_observable <- 5
#'
#' # generate a square matrix of transition probabilities in the hidden states
#' transition <- simulate_simplex_matrix(n_hidden, n_hidden)
#'
#' # and a rectangular matrix of probabilities of each observable state,
#' # given each hidden state
#' emission <- simulate_simplex_matrix(n_hidden, n_observable)
#'
#' # simulate an HMM for 10 timesteps
#' hmm_data <- simulate_hmm(transition, emission, 10)
#'
#' # pull out the observed states
#' hmm_data$observed
#'
simulate_hmm <- function (transition = random_simplex_matrix(3, 3),
                          emission = random_simplex_matrix(3, 5),
                          n_timesteps = 10) {

  # number of hidden and observedstates
  n_transition <- nrow(transition)
  n_observed <- ncol(emission)

  # check the number of timesteps
  n_timesteps <- as.integer(n_timesteps)

  if (!is.vector(n_timesteps) ||
      length(n_timesteps) != 1 ||
      n_timesteps < 1L) {

    stop ("n_timesteps must be a postive scalar integer",
          call. = FALSE)

  }

  if (inherits(transition, "greta_array") ||
      inherits(emission, "greta_array")) {
    stop ("transition and emission must be R matrices, not greta arrays",
          call. = FALSE)

  }

  # check the dimensions
  if (length(dim(transition)) != 2L ||
      ncol(transition) != n_transition) {

    stop ("transition must be a square matrix ",
          "but has dimensions ",
          paste(dim(transition), collapse = " x "),
          call. = FALSE)

  }

  if (length(dim(emission)) != 2L ||
      nrow(emission) != n_transition) {

    stop ("emission must be a matrix with ",
          "the same number of rows as transition (", n_transition, ") ",
          "but has dimensions ",
          paste(dim(emission), collapse = " x "),
          call. = FALSE)

  }


  # get the hidden states
  hidden_states <- rep(NA, n_timesteps)
  hidden_states[1] <- get_initial_state(transition)
  for (t in 2:n_timesteps)
    hidden_states[t] <- get_next_state(hidden_states[t - 1], transition)

  # corresponding observed states
  probs <- emission[hidden_states, ]
  observed_states <- apply(probs, 1,
                           function (p) {
                             sample.int(n_observed, 1, prob = p)
                           })

  # return the data as a list
  list(hidden = hidden_states,
       observed = observed_states,
       transition = transition,
       emission = emission)
}

#' @rdname simulate_hmm
#'
#' @export
#'
#' @param nrow,ncol the number of rows and columns in the matrix
random_simplex_matrix <- function (nrow, ncol) {

  n_elem <- nrow * ncol
  elems <- runif(n_elem)
  matrix <- matrix(elems, nrow, ncol)
  sums <- rowSums(matrix)
  sweep(matrix, 1, sums, "/")

}

# simulate the initial hidden state, with the selection probability at the
# mean-field (iterate until convergence)
get_initial_state <- function (transition, max_iterations = 100, tolerance = 0.001) {
  n_transition <- nrow(transition)
  it <- 0
  converged <- FALSE
  init <- runif(n_transition)
  init <- init / sum(init)
  while (it < max_iterations & !converged) {
    init_old <- init
    init <- init %*% transition
    init <- init / sum(init)
    converged <- all(abs(init - init_old) < tolerance)
  }
  sample.int(n_transition, 1, prob = init)
}

get_next_state <- function (state, transition) {
  n_transition <- nrow(transition)
  sample.int(n_transition, 1, prob = transition[state, ])
}



