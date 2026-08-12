import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
import losses
import dataset

def predict(
        model,
        clean_test_dataset,
        poison_test_dataset,
        num_workers,
        device,
        dataset_name,
        attack_type,
        split_time,
        logger
):
    clean_test_data_loader = DataLoader(clean_test_dataset, batch_size=128, num_workers=num_workers,
                                        pin_memory=True,
                                        shuffle=False)
    poisoned_test_data_loader = DataLoader(poison_test_dataset, batch_size=128, num_workers=num_workers,
                                           pin_memory=True,
                                           shuffle=False)
    criterion = losses.SCELoss(num_classes=dataset.get_num_classes(dataset_name, split_time),
                               reduction='none')
    model.eval()
    correct_clean = 0
    correct_poisoned = 0
    total_clean = 0
    total_poisoned = 0
    total_loss_clean = 0
    total_loss_poisoned = 0
    predict_result = []
    with torch.no_grad():
        for image_path_clean, inputs_clean, labels_clean in clean_test_data_loader:
            inputs_clean, labels_clean = inputs_clean.to(device), labels_clean.to(device)
            outputs = model(inputs_clean)
            loss = criterion(outputs, labels_clean)
            total_loss_clean += loss.sum().item()
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(inputs_clean)):
                row = [
                    attack_type,
                    int(labels_clean[i]),
                    float(loss[i]),
                    image_path_clean[i],
                    predicted[i]
                ]
                predict_result.append(row)
            _, predicted_clean = torch.max(outputs.data, 1)
            total_clean += labels_clean.size(0)
            correct_clean += (predicted_clean == labels_clean).sum().item()
        logger.info(f"clean ACC : {correct_clean / total_clean}")
        logger.info(f"Average Loss for Clean Data: {total_loss_clean / total_clean}")

        for image_path_poison,inputs_poison, labels_poison in poisoned_test_data_loader:
            inputs_poison, labels_poison = inputs_poison.to(device), labels_poison.to(device)
            outputs = model(inputs_poison)
            loss = criterion(outputs, labels_poison)
            total_loss_poisoned += loss.sum().item()
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(inputs_poison)):
                row = [
                    attack_type,
                    int(labels_poison[i]),
                    float(loss[i]),
                    image_path_poison[i],
                    predicted[i]
                ]
                predict_result.append(row)
            _, predicted_poisoned = torch.max(outputs.data, 1)
            total_poisoned += labels_poison.size(0)
            correct_poisoned += (predicted_poisoned == labels_poison).sum().item()
        logger.info(f"poison ACC : {correct_poisoned / total_poisoned}")
        logger.info(f"Average Loss for Poisoned Data: {total_loss_poisoned / total_poisoned}")

    return predict_result,correct_clean / total_clean,correct_poisoned / total_poisoned