from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def get_dataset_targets(dataset: Dataset) -> List[int]:
    """
    Extract class labels from a Dataset or nested Subset.
    """

    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)

        return [
            int(parent_targets[index])
            for index in dataset.indices
        ]

    if hasattr(dataset, "targets"):
        return [
            int(target)
            for target in dataset.targets
        ]

    if hasattr(dataset, "labels"):
        return [
            int(label)
            for label in dataset.labels
        ]

    raise AttributeError(
        "The dataset does not provide targets or labels."
    )


def split_dataset_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Split a dataset approximately equally among clients using IID sharding.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero.")

    if num_clients > len(dataset):
        raise ValueError(
            "num_clients cannot be greater than the number of samples."
        )

    generator = torch.Generator().manual_seed(seed)

    shuffled_indices = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    client_indices = [
        [] for _ in range(num_clients)
    ]

    for position, sample_index in enumerate(shuffled_indices):
        client_id = position % num_clients
        client_indices[client_id].append(sample_index)

    return [
        Subset(dataset, indices)
        for indices in client_indices
    ]


def split_dataset_non_iid(
    dataset: Dataset,
    num_clients: int,
    classes_per_client: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Split the dataset among clients so that each client receives samples
    from approximately `classes_per_client` different classes.

    This simulates statistical heterogeneity.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero.")

    targets = get_dataset_targets(dataset)

    unique_classes = sorted(set(targets))
    num_classes = len(unique_classes)

    if classes_per_client <= 0:
        raise ValueError(
            "classes_per_client must be greater than zero."
        )

    if classes_per_client > num_classes:
        raise ValueError(
            "classes_per_client cannot exceed the number of classes."
        )

    generator = torch.Generator().manual_seed(seed)

    class_to_indices = {
        class_id: []
        for class_id in unique_classes
    }

    for sample_index, class_id in enumerate(targets):
        class_to_indices[class_id].append(sample_index)

    for class_id in unique_classes:
        indices = class_to_indices[class_id]

        permutation = torch.randperm(
            len(indices),
            generator=generator,
        ).tolist()

        class_to_indices[class_id] = [
            indices[position]
            for position in permutation
        ]

    client_indices = [
        [] for _ in range(num_clients)
    ]

    class_assignments = [
        [] for _ in range(num_clients)
    ]

    total_assignments = num_clients * classes_per_client

    repeated_classes = []

    while len(repeated_classes) < total_assignments:
        class_permutation = torch.randperm(
            num_classes,
            generator=generator,
        ).tolist()

        repeated_classes.extend(
            unique_classes[position]
            for position in class_permutation
        )

    repeated_classes = repeated_classes[:total_assignments]

    assignment_index = 0

    for client_id in range(num_clients):
        assigned_classes = repeated_classes[
            assignment_index:
            assignment_index + classes_per_client
        ]

        class_assignments[client_id] = assigned_classes
        assignment_index += classes_per_client

    class_to_clients = {
        class_id: []
        for class_id in unique_classes
    }

    for client_id, assigned_classes in enumerate(class_assignments):
        for class_id in assigned_classes:
            class_to_clients[class_id].append(client_id)

    for class_id, indices in class_to_indices.items():
        assigned_clients = class_to_clients[class_id]

        if not assigned_clients:
            assigned_clients = [
                class_id % num_clients
            ]

        for position, sample_index in enumerate(indices):
            client_id = assigned_clients[
                position % len(assigned_clients)
            ]

            client_indices[client_id].append(sample_index)

    for client_id in range(num_clients):
        if not client_indices[client_id]:
            raise RuntimeError(
                f"Client {client_id} received no samples."
            )

        permutation = torch.randperm(
            len(client_indices[client_id]),
            generator=generator,
        ).tolist()

        client_indices[client_id] = [
            client_indices[client_id][position]
            for position in permutation
        ]

    return [
        Subset(dataset, indices)
        for indices in client_indices
    ]


def build_client_dataloaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    num_workers: int = 2,
    partition: str = "iid",
    classes_per_client: Optional[int] = None,
    seed: int = 42,
) -> List[DataLoader]:
    """
    Build one DataLoader for each federated client.

    partition:
        - "iid"
        - "non_iid"
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if partition == "iid":
        client_datasets = split_dataset_iid(
            dataset=dataset,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition == "non_iid":
        if classes_per_client is None:
            raise ValueError(
                "classes_per_client is required for non-IID partitioning."
            )

        client_datasets = split_dataset_non_iid(
            dataset=dataset,
            num_clients=num_clients,
            classes_per_client=classes_per_client,
            seed=seed,
        )

    else:
        raise ValueError(
            "partition must be either 'iid' or 'non_iid'."
        )

    client_loaders = []

    for client_id, client_dataset in enumerate(client_datasets):
        print(
            f"Client {client_id}: "
            f"{len(client_dataset)} samples"
        )

        client_loaders.append(
            DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )

    return client_loaders