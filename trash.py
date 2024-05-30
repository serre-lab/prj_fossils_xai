import helpers
from fossils_processing import parse_fossils, get_fossils_paths


if __name__ == "__main__":
    # model_path = './models/model-RELU.h5'
    # model, g, h = helpers.get_model(model_path)

    # # print(g.summary())
    # # print(h.summary())
    # print(model.summary())

    csv_path = "./csv/fossils.csv"
    class_names, class_labels = helpers.get_families(csv_path)

    print(class_names)
    print(class_labels)

    class_to_id = {c: class_labels[i] for i, c in enumerate(class_names)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    print(class_to_id)
    print(id_to_class)

    data_dir = (
        "/cifs/data/tserre_lrs/projects/prj_fossils/data/2024/Florissant_Fossil_v2.0"
    )
    x_fossils = get_fossils_paths(class_names, class_labels, data_dir)
    print(len(x_fossils.keys()))
