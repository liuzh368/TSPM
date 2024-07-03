import numpy as np

def check_files(savedir, entity_filename, relation_dict_filename, relation_files_base, timeslots):
    # 检查实体文件
    entity_file_path = f'{savedir}/{entity_filename}.npy'
    entities = np.load(entity_file_path, allow_pickle=True)
    print(f'Entities ({entity_file_path}):')
    print(entities)

    # 检查关系字典文件
    relation_dict_file_path = f'{savedir}/{relation_dict_filename}.npy'
    relation_dict = np.load(relation_dict_file_path, allow_pickle=True)
    print(f'Relation dict ({relation_dict_file_path}):')
    print(relation_dict)

    # 检查每个时间段的关系文件
    for timeslot in timeslots:
        relation_file_path = f'{savedir}/{relation_files_base}_{timeslot}.npy'
        relations = np.load(relation_file_path, allow_pickle=True)
        print(f'Relations for timeslot {timeslot} ({relation_file_path}):')
        print(relations)

# 调用函数来检查文件
check_files(savedir='../WORK',
            entity_filename='entity_list_gowalla',
            relation_dict_filename='relation_dict_gowalla',
            relation_files_base='relation_only_pre_and_sub_gowalla',
            timeslots=range(3))
