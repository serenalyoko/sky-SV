import os
import json
import pandas as pd


def write_to_file(out_edges, out_nodes, edge_dict, nodes):
    f_edges = open(out_edges, "w")
    f_nodes = open(out_nodes, "w")

    node_header = "node\n"
    f_nodes.write(node_header)
    for n in nodes:
        line = n + "\n"
        f_nodes.write(line)
    f_nodes.close()

    edge_header = "source,target,edge_weight\n"
    f_edges.write(edge_header)
    for i in range(len(edge_dict["source"])):
        line = str(edge_dict["source"][i]) + "," + str(edge_dict["target"][i]) + ","+ str(edge_dict["edge_weight"][i]) + "\n"
        f_edges.write(line)
    f_edges.close()


def construct_network(data_dir, edge_var, drop_edge_condition):
    edge_dict = {"source":[], "target":[], "edge_weight":[]}
    nodes = set()

    files = os.listdir(data_dir)
    for file in files:
        file_path = os.path.join(data_dir, file)
        print("reading", file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            print("File has", len(df["random_id1"]), "rows." )
            all_source = df["random_id0"]
            all_target = df["random_id1"]
            for i in range(len(all_source)):
                nodes.add(all_source[i])
                nodes.add(all_target[i])

            if len(drop_edge_condition["column"]) >= 0:
                for i in range(len(drop_edge_condition["column"])):
                    df = df[df[drop_edge_condition["column"][i]] != drop_edge_condition["value"][i]]
            print("File has", len(df["random_id1"]), "rows after edge removal. ")

            source = df["random_id0"]
            target = df["random_id1"]
            edge_weight = df[edge_var]
            for i in range(len(source)):
                edge_dict["source"].append(source.iloc[i])
                edge_dict["target"].append(target.iloc[i])
                edge_dict["edge_weight"].append(edge_weight.iloc[i])

    print("Done reading all the files. Now exporting to csvs.")
    write_to_file("edge_list.csv", "node_list.csv", edge_dict,nodes)

    return edge_dict,nodes

def get_social_features(all_features_file, nodes, out_file ):
    df = pd.read_csv(all_features_file)
    df = df.drop(df[df.random_id not in nodes].index)
    df.to_csv(out_file)




if __name__=="__main__":
    """
    If you don't see circumstances where you need to exclude an edge, use the following instead.
    default: drop_edge_condition = {"column":[], "value": [] }
    """
    drop_edge_condition = {"column":["playtime_1on1"], "value": [0] }
    edge_dict, nodes = construct_network("/Users/siyizhou/Documents/2024Fall/pythonProject/network_data", "playtime_1on1", drop_edge_condition)
    get_social_features("socia_features.csv", nodes, "filtered_social featurew")