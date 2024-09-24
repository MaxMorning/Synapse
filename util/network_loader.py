import torch


def load_network(network_type, weight_path):
    network = None

    if network_type == 'Restormer':
        from arch.Restormer.restormer_arch import Restormer
        network = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias"
        )

    else:
        raise NotImplementedError(f'{network_type} is not implemented')

    network.load_state_dict(torch.load(weight_path))

    return network
