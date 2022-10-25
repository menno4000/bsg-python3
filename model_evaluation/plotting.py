import os
import matplotlib.pyplot as plt

from .bsg_measure_functions import read_cos_from_entailment_results_file, read_kl_from_entailment_results_file, read_word_count_dict_from_file

# plots the best 10 entailment measurement results for a given entailment results file
def plot_single_best(target_word, entailment_input_data_path, output_base_path, plot_name='Kullback Leibler Divergence'):
    print(f"plotting best {plot_name} entailment results for target word {target_word}")
    if plot_name == 'Kullback Leibler Divergence':
        single_plot_data = read_kl_from_entailment_results_file(entailment_input_data_path)
    elif plot_name == 'Cosine Similarity':
        single_plot_data = read_cos_from_entailment_results_file(entailment_input_data_path)
    single_plot_data_sorted = sorted(single_plot_data, key=lambda x: x[1])

    if plot_name == 'Kullback Leibler Divergence':
        single_plot_data_best = single_plot_data_sorted[:10]

    elif plot_name == 'Cosine Similarity':
        single_plot_data_best = single_plot_data_sorted[-10:]

    plt.clf()
    plt.plot([d[1] for d in single_plot_data_best])

    plot_name_abbreviation = 'kl' if plot_name == 'Kullback Leibler Divergence' else 'cos'
    plot_filename = f"{target_word}_best_{plot_name_abbreviation}.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"{plot_name} for the Word {target_word}\n(Best 10)")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, pad_inches=0)

# plots the complete curve of an entailment measurement for a given entailment results file
def plot_single_all(target_word, entailment_input_data_path, output_base_path, plot_name='Kullback Leibler Divergence'):
    print(f"plotting {plot_name} entailment results for target word {target_word}")
    plot_name_abbreviation = ''
    data = []
    if plot_name == 'Kullback Leibler Divergence':
        plot_name_abbreviation = 'kl'
        data = read_kl_from_entailment_results_file(entailment_input_data_path)
    elif plot_name == 'Cosine Similarity':
        plot_name_abbreviation = 'cos'
        data = read_cos_from_entailment_results_file(entailment_input_data_path)
    data_sorted = sorted(data, key=lambda x: x[1])

    plt.clf()
    plt.plot([d[1] for d in data_sorted])

    plot_filename = f"{target_word}_{plot_name_abbreviation}.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"{plot_name} for the Word {target_word}\n")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, pad_inches=0)


# plots set of entailment results categorized based on previously gathered part of speech tagging
def plot_multi_pos_tags(entailment_results_base_path, output_base_path, target_word_tuples, plot_name='Kullback Leibler Divergence'):
    entailment_results_files = os.listdir(entailment_results_base_path)
    for target_word_tuple in target_word_tuples:
        plt.clf()
        target_word = target_word_tuple[0]
        print(f"plotting best {plot_name} entailment results by part of speech tags for target word {target_word}")
        target_category = target_word_tuple[1]
        relevant_entailment_result_dirs = [directory for directory in entailment_results_files if directory.startswith(target_category)]
        for entailment_result_dir in relevant_entailment_result_dirs:
            reference_category = entailment_result_dir.split('_')[-1]
            entailment_results_filename = entailment_results_base_path + entailment_result_dir + '/' + target_word + '_entailment_results.csv'
            if plot_name == 'Kullback Leibler Divergence':
                data = read_kl_from_entailment_results_file(entailment_results_filename)
            else:
                data = read_cos_from_entailment_results_file(entailment_results_filename)
            data_sorted = sorted(data, key=lambda x: x[1])
            data_sorted_subset = data_sorted[:10] if plot_name == 'Kullback Leibler Divergence' else data_sorted[-10:]
            if plot_name != 'Kullback Leibler Divergence':
                data_sorted_subset.reverse()
            plt.plot([d[1] for d in data_sorted_subset], label=reference_category)
            print(f"{reference_category}: {[ds[0] for ds in data_sorted_subset]}")

        if plot_name == 'Kullback Leibler Divergence':
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper right')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.title(f"{plot_name} for the Word '{target_word}'\ncompared with different POS-grouped lists (Best 10)")
        plot_name_abbreviation = 'kl' if plot_name == 'Kullback Leibler Divergence' else 'cos'
        plot_filename = f"{target_category}_{target_word}_{plot_name_abbreviation}_pos_tag.png"

        plot_path = output_base_path + plot_filename
        print(f"saving figure {plot_filename} to {plot_path}")
        plt.savefig(plot_path, pad_inches=0)


# plots available entailment results categorized based on previously gathered part of speech tagging
def plot_multi_pos_tags_mean(entailment_results_base_path, output_base_path, size, plot_name='Kullback Leibler Divergence'):
    entailment_results_files = os.listdir(entailment_results_base_path)
    pos_category_dirs = {}
    for entailment_results_dir in entailment_results_files:
        pos_category = entailment_results_dir.split('_')[0]
        if pos_category not in pos_category_dirs.keys():
            pos_category_dirs[pos_category] = [entailment_results_dir]
        else:
            pos_category_dirs[pos_category].append(entailment_results_dir)

    for pos_category in pos_category_dirs.keys():
        plt.clf()
        entailment_results = {}
        print(f"plotting mean best entailment data grouped by part of speech tag for category {pos_category}")
        for entailment_results_dir in pos_category_dirs[pos_category]:
            reference_category = entailment_results_dir.split('_')[-1]
            entailment_results[reference_category] = []
            entailment_results_files = os.listdir(os.path.join(entailment_results_base_path, entailment_results_dir))
            for entailment_results_file in entailment_results_files:
                entailment_results_filename = entailment_results_base_path + entailment_results_dir + '/' + entailment_results_file
                if plot_name == 'Kullback Leibler Divergence':
                    data = read_kl_from_entailment_results_file(entailment_results_filename)
                else:
                    data = read_cos_from_entailment_results_file(entailment_results_filename)
                if len(data) > 0:
                    data_sorted = sorted(data, key=lambda x: x[1])
                    data_sorted_subset = data_sorted[:size] if plot_name == 'Kullback Leibler Divergence' else data_sorted[-size:]
                    if plot_name != 'Kullback Leibler Divergence':
                        data_sorted_subset.reverse()
                    entailment_results[reference_category].append(data_sorted_subset)
            print(f"collected {len(entailment_results[reference_category])} entailment datasets for reference category {reference_category}")

        for reference_category in entailment_results.keys():
            data_mean = [d[1] for d in entailment_results[reference_category][0]]
            for data in entailment_results[reference_category]:
                for i in range(0, len(data_mean)):
                    new_value = (float(data_mean[i]) + float(data[i][1]))/2
                    data_mean[i] = new_value
            plt.plot(data_mean, label=reference_category)

        if plot_name == 'Kullback Leibler Divergence':
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper right')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.title(f"{plot_name} for the POS-Category '{pos_category}'\ncompared with different POS-grouped lists (Best 10)")
        plot_name_abbreviation = 'kl' if plot_name == 'Kullback Leibler Divergence' else 'cos'
        plot_filename = f"{pos_category}_{plot_name_abbreviation}_mean_pos_tag.png"

        plot_path = output_base_path + plot_filename
        print(f"saving figure {plot_filename} to {plot_path}")
        plt.savefig(plot_path, pad_inches=0)


# plot both kl and cosine sim in same plot
def plot_single_all_both_modes(target_word, entailment_input_data_path, output_base_path):
    print(f"plotting entailment results of all modes for target word {target_word}")
    data_kl = read_kl_from_entailment_results_file(entailment_input_data_path)
    data_cos = read_cos_from_entailment_results_file(entailment_input_data_path)
    data_kl_sorted = sorted(data_kl, key=lambda x: x[1])
    data_cos_sorted = sorted(data_cos, key=lambda x: x[1])
    data_cos_sorted.reverse()

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel("Kullblack Leibler Divergence")
    plot_part.set_ylabel("Cosine Similarity")
    p1, = host.plot([d[1] for d in data_kl_sorted], label="KL", color="C0")
    p2, = plot_part.plot([d[1] for d in data_cos_sorted], label="Cosine Sim", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='upper center')

    plot_filename = f"{target_word}_both.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Sorted Entailment Results of Both Modes for the Word {target_word}\n")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, pad_inches=0)

# plot both kl and cosine sim in same plot
def plot_available_both_modes(entailment_input_base_data_path, output_base_path):
    entailment_results_files = os.listdir(entailment_input_base_data_path)
    if len(entailment_results_files) == 0:
        print("no entailment data to plot")
        return
    print(f"plotting mean results of both modes for {len(entailment_results_files)} available entailment results")

    entailment_data_kl = []
    entailment_data_cos = []
    min_index = 0
    min_length = 0
    index = 0
    for entailment_results_file in entailment_results_files:
        entailment_input_data_path = os.path.join(entailment_input_base_data_path, entailment_results_file)
        entailment_data_kl_current = read_kl_from_entailment_results_file(entailment_input_data_path)
        entailment_data_cos_current = read_cos_from_entailment_results_file(entailment_input_data_path)
        if len(entailment_data_kl_current) > 0:
            entailment_data_kl_current_sorted = sorted(entailment_data_kl_current, key=lambda x: x[1])
            entailment_data_cos_current_sorted = sorted(entailment_data_cos_current, key=lambda x: x[1])
            entailment_data_cos_current_sorted.reverse()
            if index > 0:
                if len(entailment_data_kl_current_sorted) < min_length:
                    min_length = len(entailment_data_kl_current_sorted)
                    min_index = index
            else:
                min_length = len(entailment_data_kl_current_sorted)
                print(f"updating min index ({min_index}) with new value {index - 1}")
                min_index = index

            index += 1
            entailment_data_kl.append(entailment_data_kl_current_sorted)
            entailment_data_cos.append(entailment_data_cos_current_sorted)

    entailment_kl_data_mean = entailment_data_kl[min_index]
    entailment_cos_data_mean = entailment_data_cos[min_index]
    for index, ed in enumerate(entailment_data_kl):
        for i in range(0, len(entailment_kl_data_mean)):
            if len(ed[i]) > 1:
                old_kl_mean_value = entailment_kl_data_mean[i][1]
                new_kl_entailment_value = ed[i][1]
                updated_kl_mean_value = (old_kl_mean_value + new_kl_entailment_value) / 2
                entailment_kl_data_mean[i][1] = updated_kl_mean_value
                old_cos_mean_value = entailment_cos_data_mean[i][1]
                new_cos_value = entailment_data_cos[index][i][1]
                updated_cos_mean_value = (old_cos_mean_value + new_cos_value) / 2
                entailment_cos_data_mean[i][1] = updated_cos_mean_value

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel("Kullblack Leibler Divergence")
    plot_part.set_ylabel("Cosine Similarity")
    p1, = host.plot([d[1] for d in entailment_kl_data_mean], label="KL", color="C0")
    p2, = plot_part.plot([d[1] for d in entailment_cos_data_mean], label="Cosine Sim", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    plot_filename = f"mean_both_sorted.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Sorted Results of Both Modes\nfrom all available Entailment Measurements")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, bbox_inches='tight')


# plot both kl and cosine sim in same plot
def plot_single_all_both_modes_unsorted(target_word, entailment_input_data_path, output_base_path):
    print(f"plotting entailment results of all modes for target word {target_word}")
    data_kl = read_kl_from_entailment_results_file(entailment_input_data_path)
    data_cos = read_cos_from_entailment_results_file(entailment_input_data_path)
    data_kl_subset = data_kl[:10]
    data_cos_subset = data_cos[:10]
    print(f"kl: {[d[0] for d in data_kl_subset]}")
    print(f"cos: {[d[0] for d in data_cos_subset]}")

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel("Kullblack Leibler Divergence")
    plot_part.set_ylabel("Cosine Similarity")
    p1, = host.plot([d[1] for d in data_kl_subset], label="KL", color="C0")
    p2, = plot_part.plot([d[1] for d in data_cos_subset], label="Cosine Sim", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    plot_filename = f"{target_word}_both_unsorted.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Unsorted Entailment Results of Both Modes for the Word {target_word}\n")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, bbox_inches='tight')


# plot sorted entailment results of single word together with word occurence amounts of reference words
def plot_single_all_single_modes_counted(target_word, entailment_input_data_path, target_count_list,  output_base_path, plot_name='Kullback Leibler Divergence'):
    print(f"plotting {plot_name} entailment results for target word {target_word}")
    plot_name_abbreviation = ''
    data = []
    if plot_name == 'Kullback Leibler Divergence':
        plot_name_abbreviation = 'kl'
        data = read_kl_from_entailment_results_file(entailment_input_data_path)
    elif plot_name == 'Cosine Similarity':
        plot_name_abbreviation = 'cos'
        data = read_cos_from_entailment_results_file(entailment_input_data_path)
    data_sorted = sorted(data, key=lambda x: x[1])

    print(f"gathering word count data matching the entailment result functions")
    word_counts = read_word_count_dict_from_file(target_count_list)
    data_wc = [word_counts[d[0]] for d in data_sorted]

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part1 = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel(plot_name)
    plot_part1.set_ylabel("Number of Word Occurences")
    p1, = host.plot([d[1] for d in data_sorted], label="KL", color="C0")
    p2, = plot_part1.plot(data_wc, label="Word Occurences", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='upper center')

    plot_filename = f"{target_word}_{plot_name_abbreviation}_counted.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Sorted Entailment Results ({plot_name}) for the Word {target_word}\n together with the words occurences")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, bbox_inches='tight')


# plot sorted entailment results of single word together with word occurence amounts of reference words
def plot_available_single_mode_counted(entailment_input_base_data_path, target_count_list,  output_base_path, plot_name='Kullback Leibler Divergence'):
    entailment_results_files = os.listdir(entailment_input_base_data_path)
    if len(entailment_results_files) == 0:
        print("no entailment data to plot")
        return
    print(f"plotting {plot_name} of {len(entailment_results_files)} available entailment results with median word occurences")

    word_counts = read_word_count_dict_from_file(target_count_list)
    plot_name_abbreviation = ''
    if plot_name == 'Kullback Leibler Divergence':
        plot_name_abbreviation = 'kl'
    elif plot_name == 'Cosine Similarity':
        plot_name_abbreviation = 'cos'
    entailment_data = []
    word_count_data = []
    min_index = 0
    min_length = 0
    index = 0
    for entailment_results_file in entailment_results_files:
        entailment_input_data_path = os.path.join(entailment_input_base_data_path, entailment_results_file)
        entailment_data_current = []
        if plot_name == 'Kullback Leibler Divergence':
            entailment_data_current = read_kl_from_entailment_results_file(entailment_input_data_path)
        elif plot_name == 'Cosine Similarity':
            entailment_data_current = read_cos_from_entailment_results_file(entailment_input_data_path)
        if len(entailment_data_current) > 0:
            entailment_data_current_sorted = sorted(entailment_data_current, key=lambda x: x[1])
            if index > 0:
                if len(entailment_data_current_sorted) < min_length:
                    min_length = len(entailment_data_current_sorted)
                    min_index = index
            else:
                min_length = len(entailment_data_current_sorted)
                min_index = index
            if plot_name == 'Cosine Similarity':
                entailment_data_current_sorted.reverse()
            entailment_data_wc = [word_counts[d[0]] for d in entailment_data_current_sorted]
            index += 1
            word_count_data.append(entailment_data_wc)
            entailment_data.append(entailment_data_current_sorted)
    entailment_data_mean = entailment_data[min_index]
    word_count_data_mean = word_count_data[min_index]

    for index, ed in enumerate(entailment_data):
        for i in range(0, len(entailment_data_mean)):
            if len(ed[i]) > 1:
                old_mean_value = entailment_data_mean[i][1]
                new_entailment_value = ed[i][1]
                updated_mean_value = (old_mean_value + new_entailment_value) / 2
                entailment_data_mean[i][1] = updated_mean_value
                old_wc_mean_value = word_count_data_mean[i]
                new_wc_value = word_count_data[index][i]
                updated_wc_mean_value = (old_wc_mean_value + new_wc_value) / 2
                word_count_data_mean[i] = updated_wc_mean_value

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part1 = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel(plot_name)
    plot_part1.set_ylabel("Number of Word Occurences")
    p1, = host.plot([d[1] for d in entailment_data_mean], label="KL", color="C0")
    p2, = plot_part1.plot(word_count_data_mean, label="Word Occurences", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='upper center')

    plot_filename = f"{plot_name_abbreviation}_counted_mean.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Sorted Entailment Results ({plot_name}) \n for the {len(entailment_data)} available Words \n together with median words occurences")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, bbox_inches='tight')


# plot both kl and cosine sim best entailment results in same plot
def plot_single_best_both_modes(target_word, entailment_input_data_path, output_base_path):
    print(f"plotting best entailment results of all modes for target word {target_word}")
    data_kl = read_kl_from_entailment_results_file(entailment_input_data_path)
    data_cos = read_cos_from_entailment_results_file(entailment_input_data_path)
    data_kl_sorted = sorted(data_kl, key=lambda x: x[1])
    data_cos_sorted = sorted(data_cos, key=lambda x: x[1])
    data_kl_best = data_kl_sorted[:10]
    data_cos_best = data_cos_sorted[-10:]
    data_cos_best.reverse()
    print(f"kl: {[d[0] for d in data_kl_best]}")
    print(f"cos: {[d[0] for d in data_cos_best]}")

    plt.clf()
    figure = plt.figure()
    host = figure.add_subplot(111)
    plot_part = host.twinx()
    host.set_xlabel("Words")
    host.set_ylabel("Kullblack Leibler Divergence")
    plot_part.set_ylabel("Cosine Similarity")
    p1, = host.plot([d[1] for d in data_kl_best], label="KL", color="C0")
    p2, = plot_part.plot([d[1] for d in data_cos_best], label="Cosine Sim", color="C1")
    lns = [p1, p2]
    host.legend(handles=lns, loc='upper center')

    plot_filename = f"{target_word}_best_both.png"
    plot_path = output_base_path + plot_filename

    plt.title(f"Best Entailment Results Both Modes for the Word {target_word}\n")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    print(f"saving figure {plot_filename} to {plot_path}")
    plt.savefig(plot_path, pad_inches=0)
