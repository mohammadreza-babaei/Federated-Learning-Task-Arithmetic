"""IID and class-constrained non-IID CIFAR-100 client partitioning."""

from typing import List, Optional

import torch

from torch.utils.data import DataLoader, Dataset, Subset


def get_dataset_targets(dataset: Dataset) -> List[int]:

    """Recover labels while respecting index mappings from nested subsets."""

    # A Subset stores indices into its parent, so recursively recover parent labels first.

    if isinstance(dataset, Subset):

        # Recover labels from the parent dataset.
        parent_targets = get_dataset_targets(dataset.dataset)

        # Keep only the labels corresponding to the indices stored in this Subset.
        return [

            int(parent_targets[index])

            for index in dataset.indices

        ]

    # torchvision datasets commonly expose labels through ``targets``.

    if hasattr(dataset, "targets"):

        # Convert all labels to regular Python integers.
        return [

            int(target)

            for target in dataset.targets

        ]

    # Some datasets, including older torchvision versions, use ``labels`` instead.

    if hasattr(dataset, "labels"):

        # Convert all labels to regular Python integers.
        return [

            int(label)

            for label in dataset.labels

        ]

    # Partitioning cannot preserve class constraints without labels.

    raise AttributeError(

        "The dataset does not provide targets or labels."

    )


def split_dataset_iid(

    dataset: Dataset,

    num_clients: int,

    seed: int = 42,

) -> List[Subset]:

    """Shuffle samples and distribute them round-robin to client shards."""

    # Invalid client counts would make the partition impossible or produce empty shards.

    if num_clients <= 0:

        raise ValueError("num_clients must be greater than zero.")

    # Prevent creating more clients than available samples.
    if num_clients > len(dataset):

        raise ValueError(

            "num_clients cannot be greater than the number of samples."

        )

    # A private generator keeps this split deterministic for the requested seed.

    generator = torch.Generator().manual_seed(seed)

    # Shuffle all dataset positions once; labels are irrelevant for IID splitting.

    shuffled_indices = torch.randperm(

        len(dataset),

        generator=generator,

    ).tolist()

    # Create one empty index list for each federated client.
    client_indices = [

        [] for _ in range(num_clients)

    ]

    # Round-robin assignment balances the number of samples across clients.

    for position, sample_index in enumerate(shuffled_indices):

        # Select the client using modulo so assignment cycles through all clients.
        client_id = position % num_clients

        # Add the shuffled sample index to that client's local shard.
        client_indices[client_id].append(sample_index)

    # Convert each client's list of indices into a PyTorch Subset.
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

    Split the dataset so each client receives samples from approximately

    `classes_per_client` different classes.

    This simulates statistical heterogeneity while preserving every sample.

    """

    # Read labels so samples can be grouped by class.

    if num_clients <= 0:

        raise ValueError("num_clients must be greater than zero.")

    # Recover the label associated with every sample in the dataset.
    targets = get_dataset_targets(dataset)

    # Find all distinct class identities and keep them ordered.
    unique_classes = sorted(set(targets))

    # Count the number of different classes available in the dataset.
    num_classes = len(unique_classes)

    # Each client must receive at least one class.
    if classes_per_client <= 0:

        raise ValueError(

            "classes_per_client must be greater than zero."

        )

    # A client cannot receive more classes than the dataset actually contains.
    if classes_per_client > num_classes:

        raise ValueError(

            "classes_per_client cannot exceed the number of classes."

        )

    # Reuse one seeded generator for every random operation in this partition.

    generator = torch.Generator().manual_seed(seed)

    # Build a class -> sample-index lookup.

    class_to_indices = {

        class_id: []

        for class_id in unique_classes

    }

    # Group every dataset sample according to its class label.
    for sample_index, class_id in enumerate(targets):

        class_to_indices[class_id].append(sample_index)

    # Shuffle samples within each class before distributing them.

    for class_id in unique_classes:

        # Retrieve all sample indices belonging to the current class.
        indices = class_to_indices[class_id]

        # Create a random permutation of positions inside this class.
        permutation = torch.randperm(

            len(indices),

            generator=generator,

        ).tolist()

        # Reorder class samples using the generated permutation.
        class_to_indices[class_id] = [

            indices[position]

            for position in permutation

        ]

    # Create one empty sample-index list for each client.
    client_indices = [

        [] for _ in range(num_clients)

    ]

    # Store which classes are assigned to each client.
    class_assignments = [

        [] for _ in range(num_clients)

    ]

    # Each client receives a target number of class assignments.

    total_assignments = num_clients * classes_per_client

    # Repeat shuffled class orders when more assignments are needed than classes.

    repeated_classes = []

    # Keep generating shuffled class orders until enough assignments exist.
    while len(repeated_classes) < total_assignments:

        # Randomly permute all available class identities.
        class_permutation = torch.randperm(

            num_classes,

            generator=generator,

        ).tolist()

        # Append the shuffled class identities to the assignment pool.
        repeated_classes.extend(

            unique_classes[position]

            for position in class_permutation

        )

    # Keep exactly the number of class assignments required.
    repeated_classes = repeated_classes[:total_assignments]

    # Give each client its requested number of class identities.

    assignment_index = 0

    for client_id in range(num_clients):

        # Select the next group of classes for the current client.
        assigned_classes = repeated_classes[

            assignment_index:

            assignment_index + classes_per_client

        ]

        # Store the class identities assigned to this client.
        class_assignments[client_id] = assigned_classes

        # Move to the next block of class assignments.
        assignment_index += classes_per_client

    # Create the reverse mapping: class -> clients allowed to receive it.
    class_to_clients = {

        class_id: []

        for class_id in unique_classes

    }

    # Reverse the mapping so each class knows which clients may receive it.

    for client_id, assigned_classes in enumerate(class_assignments):

        # A client may be assigned several different classes.
        for class_id in assigned_classes:

            # Record that this client is allowed to receive samples of this class.
            class_to_clients[class_id].append(client_id)

    # Process every class and distribute its samples among its assigned clients.
    for class_id, indices in class_to_indices.items():

        # Get the clients that were assigned this class.
        assigned_clients = class_to_clients[class_id]

        # If random assignment omitted a class, keep its samples instead of dropping them.

        if not assigned_clients:

            # Use a deterministic fallback client based on the class id.
            assigned_clients = [

                class_id % num_clients

            ]

        # Distribute samples of this class among the clients assigned to it.

        for position, sample_index in enumerate(indices):

            # Cycle through the allowed clients in round-robin order.
            client_id = assigned_clients[

                position % len(assigned_clients)

            ]

            # Add the sample to the selected client's local dataset.
            client_indices[client_id].append(sample_index)

            # Ensure every client is usable and randomize the final local sample order.

    # Verify and shuffle the final local dataset of every client.
    for client_id in range(num_clients):

        # Federated training requires every client to contain at least one sample.
        if not client_indices[client_id]:

            raise RuntimeError(

                f"Client {client_id} received no samples."

            )

        # Randomly permute the final sample order inside this client's dataset.
        permutation = torch.randperm(

            len(client_indices[client_id]),

            generator=generator,

        ).tolist()

        # Apply the permutation to the client's sample indices.
        client_indices[client_id] = [

            client_indices[client_id][position]

            for position in permutation

        ]

    # Convert every client's index list into a PyTorch Subset.
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

    Build one shuffled DataLoader for each federated client.

    partition:

        - "iid"

        - "non_iid"

    ``classes_per_client`` is required only for non-IID partitioning.

    """

    # A non-positive batch size is invalid for every partition type.

    if batch_size <= 0:

        raise ValueError("batch_size must be greater than zero.")

    if partition == "iid":

    # Select the partitioning strategy requested by the caller.

        # Build IID local datasets for all clients.
        client_datasets = split_dataset_iid(

            dataset=dataset,

            num_clients=num_clients,

            seed=seed,

        )

    elif partition == "non_iid":

        # Non-IID splitting needs to know how many classes each client may receive.

        if classes_per_client is None:

            raise ValueError(

                "classes_per_client is required for non-IID partitioning."

            )

        # Build class-constrained non-IID local datasets.
        client_datasets = split_dataset_non_iid(

            dataset=dataset,

            num_clients=num_clients,

            classes_per_client=classes_per_client,

            seed=seed,

        )

    else:

        # Reject unsupported partition names.
        raise ValueError(

            "partition must be either 'iid' or 'non_iid'."

        )

    # Wrap each client subset in its own shuffled DataLoader.

    client_loaders = []

    # Create one independent DataLoader for each federated client.
    for client_id, client_dataset in enumerate(client_datasets):

        # Print the local dataset size for inspection and debugging.
        print(

            f"Client {client_id}: "

            f"{len(client_dataset)} samples"

        )

        # Add the current client's DataLoader to the final list.
        client_loaders.append(

            DataLoader(

                # Use only this client's local subset.
                client_dataset,

                # Define the number of samples processed per local batch.
                batch_size=batch_size,

                # Shuffle local data during training.
                shuffle=True,

                # Number of worker processes used for data loading.
                num_workers=num_workers,

                # Improve CPU-to-GPU transfer when CUDA is available.
                pin_memory=torch.cuda.is_available(),

            )

        )

    # Return one DataLoader for each federated client.
    return client_loaders