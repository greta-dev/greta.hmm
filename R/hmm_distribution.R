# a distribution node and greta array constructor for hidden markov models

#' @importFrom greta .internals
distrib <- greta::.internals$nodes$constructors$distrib
as.greta_array <- greta::.internals$greta_arrays$as.greta_array
is_scalar <- greta::.internals$utils$misc$is_scalar
tf_as_integer <- greta::.internals$tensors$tf_as_integer

#' @title Define a Hidden Markov Model
#' @name hmm
#'
#' @export
#'
#' @description `hmm()` create a variable greta array following a Hidden Markov
#'   Model (HMM) 'distribution'. That is a probability distribution whose
#'   density is given by the probability of observing a sequence of observed
#'   states (the variable greta array), given a transition and an emission
#'   matrix as its parameters. This can be viewed as a compound distribution
#'   consisting of a categorical distribution over observed states, conditional
#'   on hidden states which (sequentially) follow another categorical
#'   distribution of states. The log density is calculated by analytically
#'   integrating out the hidden states using the forward algorithm.
#'
#'   This is a discrete, multivariate distribution, and is most likely to be
#'   used with \code{\link{distribution}()} to define a complete HMM, as in the
#'   example.
#'
#'   The transition and emission matrices should represent simplices; having
#'   rows summing to 1. These can be create with e.g.
#'   \code{\link[greta:imultilogit]{imultilogit}()}.
#'
#' @param initial a length K row vector (ie. 1 x K matrix) of probabilities of
#'   initial hidden states
#' @param transition a K x K matrix of transition probabilities between hidden
#'   states
#' @param emission a K x N matrix of emission probabilities between hidden and
#'   observed states
#' @param n_timesteps the number of timesteps (length of the observed state
#'   matrix) - must be a positive scalar integer
#'
#' @examples
#' \dontrun{
#'
#' # simulate data
#' n_hidden <- 2
#' n_observable <- 2
#' timesteps <- 20
#' initial <- random_simplex_matrix(n_hidden, 1)
#' transition <- random_simplex_matrix(n_hidden,
#'                                     n_hidden)
#' emission <- random_simplex_matrix(n_hidden,
#'                                   n_observable)
#' hmm_data <- simulate_hmm(initial,
#'                          transition,
#'                          emission,
#'                          timesteps)
#' obs <- hmm_data$observed
#'
#' # create simplex variables for the matrices
#' initial_raw <- uniform(0, 1, dim = c(1, n_hidden - 1))
#' initial <- imultilogit(initial_raw)
#'
#' transition_raw <- uniform(0, 1, dim = c(n_hidden, n_hidden - 1))
#' transition <- imultilogit(transition_raw)
#'
#' emission_raw <- uniform(0, 1, dim = c(n_hidden, n_observable - 1))
#' emission <- imultilogit(emission_raw)
#'
#' # define the HMM over the observed states
#' distribution(obs) <- hmm(initial, transition, emission, timesteps)
#'
#' # build and fit the model
#' m <- model(transition)
#' draws <- mcmc(m)
#'
#' # compare the posterior means with the true transitions
#' means <- summary(draws)$statistics[, 1]
#' matrix(means, n_hidden, n_hidden)
#' hmm_data$transition
#'
#' }
hmm <- function (initial, transition, emission, n_timesteps) {
  distrib("hmm", initial, transition, emission, n_timesteps)
}

#' @importFrom R6 R6Class
#' @importFrom tensorflow tf
hmm_distribution <- R6::R6Class(
  "hmm_distribution",
  inherit = greta::.internals$nodes$node_classes$distribution_node,
  public = list(

    n_timesteps = 0L,

    initialize = function (initial, transition, emission, n_timesteps) {

      # coerce things to greta arrays
      initial <- as.greta_array(initial)
      transition <- as.greta_array(transition)
      emission <- as.greta_array(emission)

      # number of transition states
      n_transition <- nrow(transition)

      # check the number of timesteps
      n_timesteps <- as.integer(n_timesteps)

      if (!is.vector(n_timesteps) ||
          length(n_timesteps) != 1 ||
          n_timesteps < 1L) {

        stop ("n_timesteps must be a postive scalar integer",
              call. = FALSE)

      }

      # check the dimensions
      if (length(dim(transition)) != 2L ||
          ncol(transition) != n_transition) {

        stop ("transition must be a square 2D greta array ",
              "but has dimensions ",
              paste(dim(transition), collapse = " x "),
              call. = FALSE)

      }

      # check the dimensions
      if (length(dim(initial)) != 2L ||
          ncol(initial) != n_transition) {

        stop ("initial must be a row vector greta array ",
              "with as many elements as hidden states ",
              "(i.e. the dimension of 'transition') ",
              "but has dimensions ",
              paste(dim(initial), collapse = " x "),
              call. = FALSE)

      }

      if (length(dim(emission)) != 2L ||
          nrow(emission) != n_transition) {

        stop ("emission must be a 2D greta array with ",
              "the same number of rows as transition (", n_transition, ") ",
              "but has dimensions ",
              paste(dim(emission), collapse = " x "),
              call. = FALSE)

      }

      super$initialize("hmm",
                       dim = c(n_timesteps, 1L),
                       discrete = TRUE)
      self$add_parameter(initial, "initial")
      self$add_parameter(transition, "transition")
      self$add_parameter(emission, "emission")

      self$n_timesteps <- n_timesteps

    },

    tf_distrib = function (parameters, dag) {

      init_prob <- parameters$initial
      trans_prob <- parameters$transition
      emiss_prob <- parameters$emission

      tfd <- tfp$distributions
      init_dist <- tfd$Categorical(probs = init_prob[, 0, ])
      trans_dist <- tfd$Categorical(probs = trans_prob)
      emiss_dist <- tfd$Categorical(probs = emiss_prob)

      hmm_dist <- tfd$HiddenMarkovModel(
        initial_distribution = init_dist,
        transition_distribution = trans_dist,
        observation_distribution = emiss_dist,
        num_steps = self$n_timesteps
      )

      log_prob <- function (x) {
        hmm_dist$log_prob(x[, , 0] - 1L)
      }

      list(log_prob = log_prob, cdf = NULL, log_cdf = NULL)

    },

    # no CDF available
    tf_cdf_function = NULL,
    tf_log_cdf_function = NULL
  )

)
