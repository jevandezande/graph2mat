from typing import Union, Type, Optional, Tuple

from pathlib import Path

from graph2mat import MatrixDataProcessor

import torch


def sanitize_checkpoint(checkpoint: dict) -> dict:
    """Makes sure that the checkpoint is compatible with the current version of e3nn_matrix."""

    checkpoint = checkpoint.copy()

    if "z_table" in checkpoint:
        checkpoint["basis_table"] = checkpoint.pop("z_table")

    data_ckpt = checkpoint["datamodule_hyper_parameters"]
    model_ckpt = checkpoint["hyper_parameters"]

    for sub_ckpt in (data_ckpt, model_ckpt):
        if "z_table" in sub_ckpt:
            sub_ckpt["basis_table"] = sub_ckpt.pop("z_table")
        if "unique_atoms" in sub_ckpt:
            sub_ckpt["unique_basis"] = sub_ckpt.pop("unique_atoms")
        if "sub_atomic_matrix" in sub_ckpt:
            sub_ckpt["sub_point_matrix"] = sub_ckpt.pop("sub_atomic_matrix")

    return checkpoint


def load_from_lit_ckpt(
    ckpt_file: Union[Path, str],
    cpu: bool = True,
    as_torch: bool = False,
) -> Tuple[torch.nn.Module, MatrixDataProcessor]:
    """Load a model from a Lightning checkpoint file.

    Parameters
    ----------
    ckpt_file : Union[Path, str]
        Path to the checkpoint file.
    cpu : bool, optional
        If True, the model is loaded on the CPU regardless of whether
        it was in the GPU when saved, by default True.
    as_torch : bool, optional
        If True, the model is returned as the bare torch.nn.Module, otherwise it is returned as a lightning module.

    Returns
    -------
    torch.nn.Module
        The model
    MatrixDataProcessor
        The processor to use for processing inputs and outputs.
    """
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    ckpt = torch.load(
        ckpt_file, map_location="cpu" if cpu else None, weights_only=False
    )

    ckpt = sanitize_checkpoint(ckpt)

    model = LitMACEMatrixModel.load_from_checkpoint(
        ckpt_file,
        basis_table=ckpt["basis_table"],
        map_location="cpu" if cpu else None,
    )

    data_processor = MatrixDataProcessor(
        out_matrix=ckpt["datamodule_hyper_parameters"]["out_matrix"],
        sub_point_matrix=ckpt["datamodule_hyper_parameters"]["sub_point_matrix"],
        symmetric_matrix=ckpt["datamodule_hyper_parameters"]["symmetric_matrix"],
        basis_table=ckpt["basis_table"],
        node_attr_getters=model.initial_node_feats,
    )

    if as_torch:
        model = model.model

    return model, data_processor
