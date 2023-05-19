import time
import numpy as np
import pandas as pd
import torch
# from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager


def print_time_dec(func):
	def wrap(*args, **kwargs):
		start = time.time()
		print(f'{func.__name__} starting!')
		result = func(*args, **kwargs)
		end = time.time()
		print(f'{func.__name__} took {np.round(end - start, 1)}s!')
		return result
	return wrap

def filter_objs(obj_list, sorted_obj_texts, obj_feats, img_feats, clip_manager, obj_topk=10, sim_threshold=0.7):
	'''
	Filter unique objects in image using cosine similarity between object embeddings.
	Input:
		obj_list: list of objects in vocabulary
		sorted_obj_texts: list of objects in image sorted by scores
		obj_feats: object embeddings
		img_feats: image embeddings
		clip_manager: clip manager
		obj_topk: number of objects to keep
		sim_threshold: cosine similarity threshold for similarity
	Output:
		filtered_objs: list of filtered objects
	'''
	# Create a dictionary that maps the objects to the cosine sim.
	obj_embeddings = dict(zip(obj_list, obj_feats))

	# Create a list that contains the objects ordered by cosine sim.
	embeddings_sorted = [obj_embeddings[w] for w in sorted_obj_texts]

	# Create a list to store the best matches
	best_matches = [sorted_obj_texts[0]]

	# Create an array to store the embeddings of the best matches
	unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

	# Loop through the 100 best objects by cosine similarity
	for i in range(1, 100):
		# Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
		max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
		# If object i is different enough to the current best matches, add it to the best matches
		if max_cos_sim < sim_threshold:
			unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
			best_matches.append(sorted_obj_texts[i])

	# Looping through the best matches, consider each terms separately by splitting the commas and spaces.
	# init clip manager
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# clip_manager = ClipManager(device=device)
	data_list = []
	for terms in best_matches:
		for term_split in terms.split(', '):
			score = clip_manager.get_image_caption_score(term_split, img_feats)
			data_list.append({
				'term': term_split, 'score': score, 'context': terms
			})
			term_split_split = term_split.split(' ')
			if len(term_split_split) > 1:
				for term_split2 in term_split_split:
					score = clip_manager.get_image_caption_score(term_split2, img_feats)
					data_list.append({
						'term': term_split2, 'score': score, 'context': terms
					})

	# Create a dataframe with the terms and scores and only keep the top term per context.
	term_df = pd.DataFrame(data_list).sort_values('score', ascending=False).drop_duplicates('context').reset_index(drop=True)

	# Prepare loop to find if additional terms can improve cosine similarity
	best_terms_sorted = term_df['term'].tolist()
	best_term = best_terms_sorted[0]
	terms_to_check = list(set(best_terms_sorted[1:]))
	best_cos_sim = term_df['score'].iloc[0]
	filtered_objs = [best_term]

	# Perform a loop to find if additional terms can improve the cosine similarity
	n_iteration = 5
	for iteration in range(n_iteration):
		data_list = []
		for term_to_test in terms_to_check:
			new_term = f"{best_term} {term_to_test}"
			score = clip_manager.get_image_caption_score(new_term, img_feats)
			data_list.append({
				'term': new_term, 'candidate': term_to_test, 'score': score
			})
		combined_df = pd.DataFrame(data_list).sort_values('score', ascending=False)
		if combined_df['score'].iloc[0] > best_cos_sim:
			best_cos_sim = combined_df['score'].iloc[0]
			filtered_objs.append(combined_df['candidate'].iloc[0])
			terms_to_check = combined_df['candidate'].tolist()[1:]
			best_term += f" {combined_df['candidate'].iloc[0]}"
		else:
			break
	# print(f'filtered terms: {filtered_objs}')

	return filtered_objs    

# alternative filtering
# alternative filtering
def filter_objs_alt(sorted_obj_texts, obj_scores, clip_manager, obj_topk=10, sim_threshold=0.7):
    '''
    Filter unique objects in image using cosine similarity between object embeddings.
    Input:
        sorted_obj_texts: list of objects in image sorted by scores
        obj_scores: clip scores for objects
        clip_manager: clip manager
        obj_topk: number of objects to keep
        sim_threshold: cosine similarity threshold for similarity
    Output:
        filtered_objs: list of filtered objects
    '''
    sorted_obj_indices = np.argsort(obj_scores)[::-1]

    unique_obj_indices = []
    # Rest of the code...

    # Extract individual words
    individual_obj_texts = []
    for obj in sorted_obj_texts:
        individual_obj_texts.extend([word.strip() for word in obj.split(',')])
    # individual_obj_texts = extract_individual_words(sorted_obj_texts)
    individual_obj_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in individual_obj_texts])

    for obj_idx in sorted_obj_indices:
        if len(unique_obj_indices) == obj_topk:
            break

        is_similar = False
        for unique_obj_idx in unique_obj_indices:
            similarity = individual_obj_feats[obj_idx].dot(individual_obj_feats[unique_obj_idx])
            if similarity >= sim_threshold:
                is_similar = True
                break

        if not is_similar:
            unique_obj_indices.append(obj_idx)

    unique_objects = [individual_obj_texts[i] for i in unique_obj_indices]
    object_list = ', '.join(unique_objects)
    # print(f'objects in image: {object_list}')

    return unique_objects
