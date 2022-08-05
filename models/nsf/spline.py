from survae.utils import sum_except_batch
from survae.transforms.bijections.functional import splines
from survae.transforms.bijections.autoregressive import AutoregressiveBijection
from einops import rearrange

from survae.nn.nets.autoregressive import MADE
from survae.transforms.bijections import Bijection


class SplineFlow(Bijection):
    def __init__(self, n_dim=3, max_nodes=29, num_bins=12, hidden_dim=64, hidden_length=4) -> None:
        super().__init__()

        self.num_bins = num_bins
        self.feature_size = n_dim * max_nodes
        self.max_nodes = max_nodes
        self.n_dim = n_dim

        self.bijector = UnconstrainedationalQuadraticSplineAutoregressiveBijection(
            autoregressive_net=MADE(self.feature_size, num_params=3*num_bins+1, hidden_features=[hidden_dim] * hidden_length),
            num_bins=num_bins,
        )

    def forward(self, x, mask=None, logs=None):
        x = rearrange(x, 'B C D -> B (C D)')
        
        z, ldj = self.bijector.forward(x)

        z = rearrange(z, 'B (C D) -> B C D', C=self.max_nodes, D=self.n_dim)

        return z, ldj

    def inverse(self, x, mask=None):
        x = rearrange(x, 'B C D -> B (C D)')

        z = self.bijector.inverse(x)

        z = rearrange(z, 'B (C D) -> B C D', C=self.max_nodes, D=self.n_dim)
        return z, 0.

class UnconstrainedationalQuadraticSplineAutoregressiveBijection(AutoregressiveBijection):

    def __init__(self, autoregressive_net, num_bins, autoregressive_order='ltr'):
        super(UnconstrainedationalQuadraticSplineAutoregressiveBijection, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]


        z, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(x,
                                                               unnormalized_widths=unnormalized_widths,
                                                               unnormalized_heights=unnormalized_heights,
                                                               unnormalized_derivatives=unnormalized_derivatives,
                                                               inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        x, _ = splines.unconstrained_rational_quadratic_spline(z,
                                                 unnormalized_widths=unnormalized_widths,
                                                 unnormalized_heights=unnormalized_heights,
                                                 unnormalized_derivatives=unnormalized_derivatives,
                                                 inverse=True)
        return x