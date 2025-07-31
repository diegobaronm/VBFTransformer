import re

def prettify_feature_names(raw_feature_names, pretty_label_dict, extra_feature_label_dict, num_particles):
    def get_particle_label(idx):
        base_labels = ['lep', 'tau', 'MET']
        jet_labels = [f'jet{i+1}' for i in range(max(0, num_particles - len(base_labels)))]
        labels = base_labels + jet_labels
        return labels[idx % len(labels)]

    pretty_features = []

    for raw in raw_feature_names:
        is_mask = raw.endswith('_nan_mask')

        # Match phi__cos(x13) or phi__sin(x14)
        m = re.match(r'(phi)__([a-z]+)\(x(\d+)\)', raw)
        if m:
            base, func, idx = m.groups()
            idx = int(idx)
            particle_label = get_particle_label(idx)
            if func == 'cos':
                label = rf"$\cos(\phi({particle_label}))$"
            elif func == 'sin':
                label = rf"$\sin(\phi({particle_label}))$"
            else:
                label = rf"{func}({base}({particle_label}))"
            if is_mask:
                label += " mask"
            pretty_features.append(label)
            continue

        # Match particle features like pt__x13
        m2 = re.match(r'([a-z]+)__x(\d+)', raw)
        if m2:
            base, idx = m2.groups()
            idx = int(idx)
            if base in pretty_label_dict:
                prefix = pretty_label_dict[base]
                particle_label = get_particle_label(idx)
                label = rf"{prefix}({particle_label})"
                if is_mask:
                    label += " mask"
                pretty_features.append(label)
                continue

        # Match extra features like opening_angle__x24
        m3 = re.match(r'([a-zA-Z_]+)__x\d+', raw)
        if m3:
            base = m3.group(1)
            if base in extra_feature_label_dict:
                label = extra_feature_label_dict[base]
                if is_mask:
                    label += " mask"
                pretty_features.append(label)
                continue

        # Match raw extra feature (no __x suffix)
        base = re.sub(r'_nan_mask$', '', raw)
        if base in extra_feature_label_dict:
            label = extra_feature_label_dict[base]
            if is_mask:
                label += " mask"
            pretty_features.append(label)
            continue

        # Fallback
        label = raw
        if is_mask:
            label += " mask"
        pretty_features.append(label)

    return pretty_features