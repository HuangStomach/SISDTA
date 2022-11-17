var tanimoto_raw = [[1.0, 0.605, 0.647, 0.539, 0.636, 0.44, 0.661, 0.609, 0.555, 0.714, 0.64, 0.737, 0.684, 0.39, 0.63, 0.685, 0.74, 0.493, 0.712, 0.449, 0.614, 0.647, 0.663, 0.662, 0.645, 0.638, 0.638, 0.551, 0.535, 0.612, 0.63, 0.546, 0.609, 0.65, 0.53, 0.529, 0.575, 0.586, 0.684, 0.542, 0.475, 0.467, 0.643, 0.507, 0.486, 0.622, 0.612, 0.566, 0.638, 0.591, 0.533, 0.575, 0.369, 0.578, 0.471, 0.613, 0.59, 0.538, 0.562, 0.566, 0.662, 0.628, 0.496, 0.516, 0.34, 0.625, 0.519, 0.577], [0.605, 1.0, 0.651, 0.432, 0.619, 0.378, 0.522, 0.628, 0.576, 0.63, 0.53, 0.66, 0.714, 0.366, 0.613, 0.639, 0.564, 0.619, 0.619, 0.445, 0.541, 0.616, 0.598, 0.47, 0.685, 0.578, 0.61, 0.55, 0.462, 0.446, 0.625, 0.541, 0.45, 0.58, 0.507, 0.612, 0.54, 0.574, 0.569, 0.563, 0.545, 0.505, 0.652, 0.532, 0.423, 0.566, 0.597, 0.608, 0.565, 0.519, 0.579, 0.559, 0.48, 0.518, 0.525, 0.598, 0.548, 0.489, 0.623, 0.546, 0.676, 0.639, 0.486, 0.535, 0.352, 0.598, 0.474, 0.469], [0.647, 0.651, 1.0, 0.459, 0.715, 0.426, 0.488, 0.729, 0.616, 0.701, 0.621, 0.658, 0.756, 0.41, 0.744, 0.713, 0.676, 0.579, 0.735, 0.496, 0.552, 0.746, 0.656, 0.513, 0.659, 0.72, 0.74, 0.652, 0.484, 0.49, 0.653, 0.5, 0.559, 0.567, 0.629, 0.583, 0.522, 0.626, 0.645, 0.652, 0.595, 0.444, 0.668, 0.553, 0.512, 0.582, 0.568, 0.673, 0.523, 0.545, 0.577, 0.524, 0.398, 0.525, 0.522, 0.737, 0.536, 0.53, 0.668, 0.662, 0.736, 0.781, 0.56, 0.564, 0.387, 0.746, 0.53, 0.569], [0.539, 0.432, 0.459, 1.0, 0.48, 0.381, 0.668, 0.543, 0.527, 0.519, 0.471, 0.468, 0.454, 0.585, 0.473, 0.419, 0.555, 0.574, 0.473, 0.521, 0.618, 0.41, 0.446, 0.583, 0.416, 0.414, 0.442, 0.599, 0.683, 0.732, 0.473, 0.498, 0.567, 0.556, 0.56, 0.572, 0.515, 0.602, 0.487, 0.615, 0.634, 0.645, 0.493, 0.592, 0.425, 0.525, 0.515, 0.383, 0.577, 0.545, 0.504, 0.486, 0.335, 0.631, 0.562, 0.603, 0.529, 0.667, 0.42, 0.433, 0.438, 0.436, 0.498, 0.559, 0.527, 0.4, 0.55, 0.571], [0.636, 0.619, 0.715, 0.48, 1.0, 0.471, 0.507, 0.685, 0.575, 0.798, 0.684, 0.682, 0.689, 0.459, 0.953, 0.762, 0.702, 0.609, 0.67, 0.486, 0.532, 0.789, 0.707, 0.485, 0.641, 0.66, 0.869, 0.602, 0.544, 0.492, 0.755, 0.517, 0.519, 0.699, 0.625, 0.555, 0.559, 0.573, 0.808, 0.615, 0.601, 0.475, 0.692, 0.578, 0.457, 0.559, 0.538, 0.691, 0.57, 0.515, 0.536, 0.524, 0.384, 0.548, 0.602, 0.698, 0.567, 0.506, 0.68, 0.61, 0.676, 0.786, 0.502, 0.515, 0.398, 0.773, 0.5, 0.565], [0.44, 0.378, 0.426, 0.381, 0.471, 1.0, 0.32, 0.443, 0.271, 0.439, 0.456, 0.403, 0.382, 0.423, 0.463, 0.43, 0.484, 0.346, 0.452, 0.357, 0.355, 0.422, 0.479, 0.347, 0.367, 0.481, 0.465, 0.34, 0.386, 0.361, 0.495, 0.397, 0.276, 0.459, 0.437, 0.32, 0.398, 0.331, 0.489, 0.327, 0.382, 0.362, 0.462, 0.359, 0.244, 0.266, 0.279, 0.408, 0.378, 0.258, 0.278, 0.25, 0.229, 0.414, 0.329, 0.504, 0.399, 0.351, 0.376, 0.411, 0.402, 0.432, 0.225, 0.265, 0.43, 0.416, 0.485, 0.328], [0.661, 0.522, 0.488, 0.668, 0.507, 0.32, 1.0, 0.53, 0.573, 0.592, 0.477, 0.532, 0.521, 0.489, 0.495, 0.52, 0.511, 0.59, 0.513, 0.493, 0.718, 0.487, 0.516, 0.631, 0.46, 0.457, 0.47, 0.674, 0.682, 0.782, 0.534, 0.568, 0.63, 0.566, 0.551, 0.708, 0.585, 0.685, 0.525, 0.674, 0.678, 0.646, 0.536, 0.615, 0.475, 0.724, 0.663, 0.458, 0.69, 0.616, 0.64, 0.67, 0.395, 0.662, 0.554, 0.555, 0.594, 0.683, 0.475, 0.464, 0.531, 0.491, 0.559, 0.653, 0.498, 0.473, 0.541, 0.596], [0.609, 0.628, 0.729, 0.543, 0.685, 0.443, 0.53, 1.0, 0.598, 0.673, 0.57, 0.616, 0.644, 0.429, 0.698, 0.622, 0.636, 0.594, 0.644, 0.48, 0.582, 0.653, 0.631, 0.484, 0.639, 0.665, 0.669, 0.701, 0.54, 0.516, 0.688, 0.528, 0.456, 0.626, 0.629, 0.624, 0.566, 0.654, 0.627, 0.665, 0.629, 0.469, 0.757, 0.557, 0.468, 0.538, 0.533, 0.647, 0.56, 0.521, 0.531, 0.507, 0.417, 0.556, 0.549, 0.69, 0.575, 0.543, 0.68, 0.64, 0.688, 0.724, 0.558, 0.533, 0.379, 0.655, 0.574, 0.518], [0.555, 0.576, 0.616, 0.527, 0.575, 0.271, 0.573, 0.598, 1.0, 0.563, 0.483, 0.579, 0.693, 0.395, 0.568, 0.578, 0.533, 0.62, 0.528, 0.546, 0.632, 0.589, 0.485, 0.588, 0.602, 0.547, 0.564, 0.691, 0.595, 0.545, 0.522, 0.555, 0.595, 0.519, 0.564, 0.641, 0.561, 0.703, 0.489, 0.715, 0.618, 0.486, 0.54, 0.647, 0.561, 0.712, 0.62, 0.489, 0.604, 0.691, 0.73, 0.686, 0.384, 0.548, 0.625, 0.541, 0.577, 0.578, 0.522, 0.615, 0.583, 0.578, 0.663, 0.599, 0.37, 0.562, 0.465, 0.67], [0.714, 0.63, 0.701, 0.519, 0.798, 0.439, 0.592, 0.673, 0.563, 1.0, 0.618, 0.752, 0.715, 0.42, 0.792, 0.775, 0.676, 0.58, 0.723, 0.492, 0.562, 0.76, 0.707, 0.55, 0.653, 0.647, 0.767, 0.608, 0.521, 0.577, 0.7, 0.533, 0.607, 0.64, 0.596, 0.566, 0.548, 0.598, 0.67, 0.622, 0.554, 0.455, 0.717, 0.55, 0.519, 0.613, 0.598, 0.673, 0.588, 0.584, 0.554, 0.587, 0.388, 0.584, 0.564, 0.634, 0.561, 0.535, 0.614, 0.629, 0.676, 0.765, 0.532, 0.622, 0.383, 0.76, 0.489, 0.577], [0.64, 0.53, 0.621, 0.471, 0.684, 0.456, 0.477, 0.57, 0.483, 0.618, 1.0, 0.621, 0.619, 0.415, 0.697, 0.713, 0.642, 0.51, 0.595, 0.405, 0.534, 0.73, 0.751, 0.486, 0.568, 0.637, 0.787, 0.515, 0.511, 0.449, 0.646, 0.468, 0.421, 0.685, 0.536, 0.481, 0.503, 0.478, 0.821, 0.496, 0.505, 0.451, 0.575, 0.458, 0.354, 0.479, 0.464, 0.596, 0.519, 0.457, 0.46, 0.461, 0.299, 0.455, 0.469, 0.65, 0.515, 0.446, 0.588, 0.582, 0.58, 0.749, 0.431, 0.412, 0.347, 0.731, 0.463, 0.561], [0.737, 0.66, 0.658, 0.468, 0.682, 0.403, 0.532, 0.616, 0.579, 0.752, 0.621, 1.0, 0.786, 0.357, 0.658, 0.805, 0.704, 0.517, 0.679, 0.468, 0.547, 0.701, 0.672, 0.566, 0.726, 0.636, 0.71, 0.533, 0.491, 0.496, 0.629, 0.541, 0.529, 0.637, 0.542, 0.503, 0.557, 0.53, 0.62, 0.565, 0.466, 0.435, 0.688, 0.531, 0.48, 0.598, 0.606, 0.623, 0.585, 0.618, 0.581, 0.596, 0.398, 0.563, 0.525, 0.601, 0.57, 0.515, 0.582, 0.62, 0.653, 0.7, 0.5, 0.513, 0.327, 0.696, 0.465, 0.544], [0.684, 0.714, 0.756, 0.454, 0.689, 0.382, 0.521, 0.644, 0.693, 0.715, 0.619, 0.786, 1.0, 0.364, 0.683, 0.797, 0.64, 0.527, 0.628, 0.482, 0.585, 0.724, 0.647, 0.504, 0.743, 0.649, 0.712, 0.602, 0.512, 0.488, 0.633, 0.53, 0.522, 0.641, 0.583, 0.565, 0.558, 0.585, 0.618, 0.642, 0.523, 0.455, 0.67, 0.577, 0.496, 0.607, 0.575, 0.633, 0.581, 0.615, 0.658, 0.574, 0.391, 0.528, 0.546, 0.675, 0.571, 0.533, 0.633, 0.651, 0.7, 0.722, 0.568, 0.523, 0.337, 0.718, 0.479, 0.538], [0.39, 0.366, 0.41, 0.585, 0.459, 0.423, 0.489, 0.429, 0.395, 0.42, 0.415, 0.357, 0.364, 1.0, 0.462, 0.378, 0.413, 0.518, 0.398, 0.497, 0.505, 0.389, 0.439, 0.437, 0.332, 0.356, 0.43, 0.517, 0.59, 0.536, 0.447, 0.447, 0.466, 0.469, 0.522, 0.48, 0.453, 0.519, 0.444, 0.514, 0.518, 0.547, 0.401, 0.543, 0.286, 0.391, 0.355, 0.343, 0.514, 0.371, 0.402, 0.354, 0.233, 0.594, 0.573, 0.52, 0.461, 0.613, 0.357, 0.387, 0.372, 0.379, 0.352, 0.412, 0.545, 0.389, 0.569, 0.458], [0.63, 0.613, 0.744, 0.473, 0.953, 0.463, 0.495, 0.698, 0.568, 0.792, 0.697, 0.658, 0.683, 0.462, 1.0, 0.742, 0.702, 0.614, 0.67, 0.473, 0.536, 0.783, 0.72, 0.483, 0.647, 0.653, 0.887, 0.62, 0.531, 0.485, 0.75, 0.49, 0.512, 0.7, 0.637, 0.559, 0.536, 0.572, 0.831, 0.621, 0.606, 0.463, 0.705, 0.566, 0.449, 0.552, 0.547, 0.697, 0.55, 0.513, 0.529, 0.528, 0.375, 0.529, 0.585, 0.698, 0.544, 0.492, 0.707, 0.642, 0.669, 0.794, 0.506, 0.519, 0.406, 0.789, 0.492, 0.595], [0.685, 0.639, 0.713, 0.419, 0.762, 0.43, 0.52, 0.622, 0.578, 0.775, 0.713, 0.805, 0.797, 0.378, 0.742, 1.0, 0.635, 0.51, 0.674, 0.471, 0.558, 0.847, 0.772, 0.504, 0.676, 0.742, 0.837, 0.562, 0.5, 0.494, 0.681, 0.529, 0.528, 0.62, 0.564, 0.523, 0.561, 0.552, 0.668, 0.576, 0.505, 0.451, 0.678, 0.536, 0.483, 0.592, 0.567, 0.688, 0.573, 0.587, 0.569, 0.578, 0.369, 0.55, 0.524, 0.607, 0.575, 0.532, 0.599, 0.653, 0.68, 0.828, 0.516, 0.5, 0.332, 0.844, 0.485, 0.519], [0.74, 0.564, 0.676, 0.555, 0.702, 0.484, 0.511, 0.636, 0.533, 0.676, 0.642, 0.704, 0.64, 0.413, 0.702, 0.635, 1.0, 0.564, 0.665, 0.43, 0.502, 0.629, 0.614, 0.578, 0.66, 0.675, 0.711, 0.524, 0.484, 0.466, 0.649, 0.424, 0.512, 0.604, 0.587, 0.504, 0.464, 0.569, 0.659, 0.533, 0.579, 0.456, 0.626, 0.551, 0.478, 0.561, 0.549, 0.614, 0.516, 0.562, 0.53, 0.541, 0.402, 0.506, 0.479, 0.66, 0.476, 0.462, 0.637, 0.582, 0.685, 0.63, 0.5, 0.508, 0.425, 0.639, 0.446, 0.614], [0.493, 0.619, 0.579, 0.574, 0.609, 0.346, 0.59, 0.594, 0.62, 0.58, 0.51, 0.517, 0.527, 0.518, 0.614, 0.51, 0.564, 1.0, 0.509, 0.541, 0.59, 0.522, 0.491, 0.558, 0.533, 0.524, 0.581, 0.684, 0.605, 0.527, 0.571, 0.484, 0.506, 0.56, 0.613, 0.747, 0.484, 0.695, 0.537, 0.677, 0.725, 0.585, 0.581, 0.659, 0.498, 0.626, 0.659, 0.531, 0.566, 0.569, 0.636, 0.631, 0.494, 0.588, 0.65, 0.561, 0.491, 0.569, 0.595, 0.526, 0.593, 0.559, 0.607, 0.664, 0.468, 0.523, 0.474, 0.594], [0.712, 0.619, 0.735, 0.473, 0.67, 0.452, 0.513, 0.644, 0.528, 0.723, 0.595, 0.679, 0.628, 0.398, 0.67, 0.674, 0.665, 0.509, 1.0, 0.486, 0.516, 0.696, 0.659, 0.546, 0.581, 0.634, 0.686, 0.55, 0.48, 0.542, 0.639, 0.512, 0.649, 0.556, 0.535, 0.528, 0.54, 0.565, 0.63, 0.547, 0.504, 0.443, 0.622, 0.47, 0.455, 0.581, 0.579, 0.581, 0.536, 0.527, 0.49, 0.531, 0.367, 0.538, 0.471, 0.61, 0.553, 0.5, 0.56, 0.603, 0.679, 0.702, 0.454, 0.558, 0.365, 0.672, 0.55, 0.575], [0.449, 0.445, 0.496, 0.521, 0.486, 0.357, 0.493, 0.48, 0.546, 0.492, 0.405, 0.468, 0.482, 0.497, 0.473, 0.471, 0.43, 0.541, 0.486, 1.0, 0.552, 0.446, 0.455, 0.507, 0.446, 0.422, 0.446, 0.557, 0.531, 0.555, 0.422, 0.519, 0.529, 0.451, 0.487, 0.538, 0.512, 0.582, 0.432, 0.574, 0.496, 0.485, 0.466, 0.481, 0.466, 0.502, 0.47, 0.45, 0.533, 0.476, 0.493, 0.443, 0.313, 0.628, 0.562, 0.479, 0.521, 0.598, 0.434, 0.469, 0.473, 0.48, 0.485, 0.537, 0.418, 0.45, 0.588, 0.457], [0.614, 0.541, 0.552, 0.618, 0.532, 0.355, 0.718, 0.582, 0.632, 0.562, 0.534, 0.547, 0.585, 0.505, 0.536, 0.558, 0.502, 0.59, 0.516, 0.552, 1.0, 0.506, 0.586, 0.64, 0.53, 0.487, 0.509, 0.735, 0.729, 0.701, 0.527, 0.688, 0.585, 0.626, 0.588, 0.705, 0.724, 0.739, 0.576, 0.75, 0.615, 0.655, 0.587, 0.615, 0.469, 0.629, 0.598, 0.463, 0.886, 0.617, 0.604, 0.621, 0.387, 0.714, 0.602, 0.579, 0.729, 0.758, 0.47, 0.528, 0.504, 0.545, 0.602, 0.561, 0.437, 0.513, 0.604, 0.618], [0.647, 0.616, 0.746, 0.41, 0.789, 0.422, 0.487, 0.653, 0.589, 0.76, 0.73, 0.701, 0.724, 0.389, 0.783, 0.847, 0.629, 0.522, 0.696, 0.446, 0.506, 1.0, 0.749, 0.492, 0.664, 0.764, 0.871, 0.584, 0.465, 0.458, 0.682, 0.493, 0.51, 0.585, 0.561, 0.525, 0.52, 0.548, 0.688, 0.573, 0.529, 0.405, 0.667, 0.509, 0.476, 0.577, 0.536, 0.703, 0.52, 0.541, 0.547, 0.545, 0.348, 0.481, 0.509, 0.638, 0.527, 0.455, 0.635, 0.69, 0.703, 0.894, 0.486, 0.5, 0.335, 0.878, 0.472, 0.545], [0.663, 0.598, 0.656, 0.446, 0.707, 0.479, 0.516, 0.631, 0.485, 0.707, 0.751, 0.672, 0.647, 0.439, 0.72, 0.772, 0.614, 0.491, 0.659, 0.455, 0.586, 0.749, 1.0, 0.5, 0.57, 0.661, 0.778, 0.558, 0.496, 0.484, 0.638, 0.531, 0.46, 0.658, 0.538, 0.52, 0.586, 0.536, 0.749, 0.549, 0.486, 0.452, 0.645, 0.464, 0.359, 0.486, 0.475, 0.705, 0.594, 0.448, 0.478, 0.468, 0.355, 0.559, 0.456, 0.654, 0.583, 0.516, 0.579, 0.573, 0.613, 0.789, 0.437, 0.453, 0.352, 0.763, 0.522, 0.51], [0.662, 0.47, 0.513, 0.583, 0.485, 0.347, 0.631, 0.484, 0.588, 0.55, 0.486, 0.566, 0.504, 0.437, 0.483, 0.504, 0.578, 0.558, 0.546, 0.507, 0.64, 0.492, 0.5, 1.0, 0.51, 0.518, 0.484, 0.59, 0.55, 0.606, 0.461, 0.554, 0.587, 0.494, 0.536, 0.562, 0.547, 0.608, 0.504, 0.584, 0.52, 0.484, 0.51, 0.517, 0.519, 0.634, 0.607, 0.433, 0.652, 0.66, 0.584, 0.624, 0.343, 0.615, 0.504, 0.484, 0.556, 0.577, 0.45, 0.507, 0.528, 0.478, 0.512, 0.541, 0.397, 0.477, 0.462, 0.646], [0.645, 0.685, 0.659, 0.416, 0.641, 0.367, 0.46, 0.639, 0.602, 0.653, 0.568, 0.726, 0.743, 0.332, 0.647, 0.676, 0.66, 0.533, 0.581, 0.446, 0.53, 0.664, 0.57, 0.51, 1.0, 0.622, 0.655, 0.539, 0.432, 0.425, 0.579, 0.488, 0.435, 0.573, 0.509, 0.502, 0.498, 0.535, 0.584, 0.566, 0.478, 0.391, 0.697, 0.498, 0.502, 0.546, 0.564, 0.628, 0.544, 0.578, 0.534, 0.538, 0.434, 0.492, 0.509, 0.563, 0.511, 0.455, 0.634, 0.632, 0.68, 0.658, 0.542, 0.506, 0.267, 0.648, 0.417, 0.479], [0.638, 0.578, 0.72, 0.414, 0.66, 0.481, 0.457, 0.665, 0.547, 0.647, 0.637, 0.636, 0.649, 0.356, 0.653, 0.742, 0.675, 0.524, 0.634, 0.422, 0.487, 0.764, 0.661, 0.518, 0.622, 1.0, 0.726, 0.563, 0.415, 0.412, 0.597, 0.412, 0.459, 0.522, 0.579, 0.502, 0.428, 0.559, 0.585, 0.54, 0.557, 0.373, 0.634, 0.504, 0.525, 0.55, 0.51, 0.646, 0.496, 0.532, 0.558, 0.51, 0.392, 0.449, 0.45, 0.602, 0.434, 0.426, 0.619, 0.668, 0.721, 0.746, 0.498, 0.475, 0.355, 0.733, 0.458, 0.522], [0.638, 0.61, 0.74, 0.442, 0.869, 0.465, 0.47, 0.669, 0.564, 0.767, 0.787, 0.71, 0.712, 0.43, 0.887, 0.837, 0.711, 0.581, 0.686, 0.446, 0.509, 0.871, 0.778, 0.484, 0.655, 0.726, 1.0, 0.584, 0.498, 0.447, 0.759, 0.472, 0.49, 0.642, 0.615, 0.527, 0.512, 0.538, 0.764, 0.574, 0.572, 0.435, 0.682, 0.54, 0.457, 0.56, 0.544, 0.725, 0.523, 0.515, 0.537, 0.535, 0.352, 0.496, 0.543, 0.668, 0.524, 0.46, 0.683, 0.672, 0.699, 0.873, 0.484, 0.498, 0.381, 0.89, 0.45, 0.598], [0.551, 0.55, 0.652, 0.599, 0.602, 0.34, 0.674, 0.701, 0.691, 0.608, 0.515, 0.533, 0.602, 0.517, 0.62, 0.562, 0.524, 0.684, 0.55, 0.557, 0.735, 0.584, 0.558, 0.59, 0.539, 0.563, 0.584, 1.0, 0.657, 0.678, 0.585, 0.57, 0.581, 0.55, 0.69, 0.851, 0.601, 0.812, 0.555, 0.897, 0.732, 0.563, 0.634, 0.639, 0.517, 0.693, 0.614, 0.56, 0.665, 0.613, 0.66, 0.617, 0.403, 0.63, 0.672, 0.622, 0.604, 0.665, 0.567, 0.594, 0.577, 0.645, 0.653, 0.651, 0.456, 0.589, 0.564, 0.637], [0.535, 0.462, 0.484, 0.683, 0.544, 0.386, 0.682, 0.54, 0.595, 0.521, 0.511, 0.491, 0.512, 0.59, 0.531, 0.5, 0.484, 0.605, 0.48, 0.531, 0.729, 0.465, 0.496, 0.55, 0.432, 0.415, 0.498, 0.657, 1.0, 0.732, 0.543, 0.613, 0.564, 0.648, 0.59, 0.632, 0.619, 0.668, 0.541, 0.65, 0.624, 0.703, 0.528, 0.646, 0.419, 0.547, 0.512, 0.392, 0.698, 0.528, 0.555, 0.506, 0.344, 0.727, 0.624, 0.642, 0.63, 0.773, 0.421, 0.464, 0.445, 0.487, 0.55, 0.556, 0.478, 0.452, 0.614, 0.59], [0.612, 0.446, 0.49, 0.732, 0.492, 0.361, 0.782, 0.516, 0.545, 0.577, 0.449, 0.496, 0.488, 0.536, 0.485, 0.494, 0.466, 0.527, 0.542, 0.555, 0.701, 0.458, 0.484, 0.606, 0.425, 0.412, 0.447, 0.678, 0.732, 1.0, 0.479, 0.596, 0.707, 0.537, 0.519, 0.644, 0.615, 0.649, 0.489, 0.67, 0.562, 0.604, 0.506, 0.545, 0.445, 0.598, 0.575, 0.39, 0.653, 0.571, 0.521, 0.547, 0.302, 0.69, 0.548, 0.529, 0.626, 0.734, 0.387, 0.47, 0.438, 0.492, 0.529, 0.616, 0.477, 0.451, 0.641, 0.547], [0.63, 0.625, 0.653, 0.473, 0.755, 0.495, 0.534, 0.688, 0.522, 0.7, 0.646, 0.629, 0.633, 0.447, 0.75, 0.681, 0.649, 0.571, 0.639, 0.422, 0.527, 0.682, 0.638, 0.461, 0.579, 0.597, 0.759, 0.585, 0.543, 0.479, 1.0, 0.528, 0.477, 0.656, 0.667, 0.58, 0.573, 0.57, 0.709, 0.582, 0.602, 0.521, 0.683, 0.643, 0.414, 0.568, 0.542, 0.648, 0.545, 0.49, 0.547, 0.514, 0.397, 0.53, 0.572, 0.669, 0.575, 0.491, 0.637, 0.593, 0.673, 0.676, 0.451, 0.485, 0.417, 0.678, 0.507, 0.534], [0.546, 0.541, 0.5, 0.498, 0.517, 0.397, 0.568, 0.528, 0.555, 0.533, 0.468, 0.541, 0.53, 0.447, 0.49, 0.529, 0.424, 0.484, 0.512, 0.519, 0.688, 0.493, 0.531, 0.554, 0.488, 0.412, 0.472, 0.57, 0.613, 0.596, 0.528, 1.0, 0.522, 0.567, 0.514, 0.553, 0.883, 0.572, 0.514, 0.603, 0.473, 0.581, 0.534, 0.514, 0.365, 0.489, 0.503, 0.432, 0.729, 0.478, 0.481, 0.476, 0.326, 0.672, 0.591, 0.518, 0.89, 0.64, 0.381, 0.483, 0.434, 0.502, 0.44, 0.445, 0.363, 0.465, 0.57, 0.479], [0.609, 0.45, 0.559, 0.567, 0.519, 0.276, 0.63, 0.456, 0.595, 0.607, 0.421, 0.529, 0.522, 0.466, 0.512, 0.528, 0.512, 0.506, 0.649, 0.529, 0.585, 0.51, 0.46, 0.587, 0.435, 0.459, 0.49, 0.581, 0.564, 0.707, 0.477, 0.522, 1.0, 0.467, 0.529, 0.555, 0.528, 0.63, 0.454, 0.591, 0.513, 0.49, 0.454, 0.542, 0.495, 0.656, 0.627, 0.418, 0.573, 0.605, 0.575, 0.601, 0.29, 0.568, 0.534, 0.466, 0.543, 0.6, 0.401, 0.513, 0.465, 0.502, 0.512, 0.629, 0.418, 0.494, 0.502, 0.637], [0.65, 0.58, 0.567, 0.556, 0.699, 0.459, 0.566, 0.626, 0.519, 0.64, 0.685, 0.637, 0.641, 0.469, 0.7, 0.62, 0.604, 0.56, 0.556, 0.451, 0.626, 0.585, 0.658, 0.494, 0.573, 0.522, 0.642, 0.55, 0.648, 0.537, 0.656, 0.567, 0.467, 1.0, 0.586, 0.535, 0.611, 0.6, 0.779, 0.581, 0.507, 0.551, 0.652, 0.586, 0.364, 0.481, 0.5, 0.529, 0.622, 0.47, 0.489, 0.469, 0.328, 0.593, 0.566, 0.679, 0.62, 0.582, 0.566, 0.529, 0.53, 0.598, 0.466, 0.445, 0.368, 0.568, 0.556, 0.532], [0.53, 0.507, 0.629, 0.56, 0.625, 0.437, 0.551, 0.629, 0.564, 0.596, 0.536, 0.542, 0.583, 0.522, 0.637, 0.564, 0.587, 0.613, 0.535, 0.487, 0.588, 0.561, 0.538, 0.536, 0.509, 0.579, 0.615, 0.69, 0.59, 0.519, 0.667, 0.514, 0.529, 0.586, 1.0, 0.618, 0.542, 0.642, 0.569, 0.668, 0.679, 0.507, 0.602, 0.664, 0.469, 0.566, 0.509, 0.529, 0.558, 0.542, 0.573, 0.51, 0.331, 0.547, 0.63, 0.64, 0.55, 0.547, 0.552, 0.621, 0.554, 0.563, 0.492, 0.5, 0.478, 0.566, 0.533, 0.627], [0.529, 0.612, 0.583, 0.572, 0.555, 0.32, 0.708, 0.624, 0.641, 0.566, 0.481, 0.503, 0.565, 0.48, 0.559, 0.523, 0.504, 0.747, 0.528, 0.538, 0.705, 0.525, 0.52, 0.562, 0.502, 0.502, 0.527, 0.851, 0.632, 0.644, 0.58, 0.553, 0.555, 0.535, 0.618, 1.0, 0.576, 0.776, 0.518, 0.82, 0.726, 0.615, 0.573, 0.631, 0.473, 0.674, 0.645, 0.517, 0.64, 0.566, 0.688, 0.651, 0.502, 0.607, 0.624, 0.583, 0.585, 0.639, 0.563, 0.529, 0.566, 0.562, 0.598, 0.686, 0.482, 0.531, 0.545, 0.592], [0.575, 0.54, 0.522, 0.515, 0.559, 0.398, 0.585, 0.566, 0.561, 0.548, 0.503, 0.557, 0.558, 0.453, 0.536, 0.561, 0.464, 0.484, 0.54, 0.512, 0.724, 0.52, 0.586, 0.547, 0.498, 0.428, 0.512, 0.601, 0.619, 0.615, 0.573, 0.883, 0.528, 0.611, 0.542, 0.576, 1.0, 0.59, 0.561, 0.628, 0.495, 0.592, 0.56, 0.52, 0.372, 0.5, 0.524, 0.46, 0.742, 0.489, 0.481, 0.487, 0.337, 0.685, 0.579, 0.534, 0.973, 0.654, 0.428, 0.506, 0.455, 0.55, 0.457, 0.456, 0.364, 0.5, 0.596, 0.49], [0.586, 0.574, 0.626, 0.602, 0.573, 0.331, 0.685, 0.654, 0.703, 0.598, 0.478, 0.53, 0.585, 0.519, 0.572, 0.552, 0.569, 0.695, 0.565, 0.582, 0.739, 0.548, 0.536, 0.608, 0.535, 0.559, 0.538, 0.812, 0.668, 0.649, 0.57, 0.572, 0.63, 0.6, 0.642, 0.776, 0.59, 1.0, 0.528, 0.782, 0.688, 0.585, 0.63, 0.709, 0.543, 0.696, 0.597, 0.516, 0.692, 0.638, 0.64, 0.634, 0.442, 0.687, 0.654, 0.591, 0.6, 0.711, 0.51, 0.548, 0.573, 0.563, 0.673, 0.624, 0.444, 0.542, 0.566, 0.64], [0.684, 0.569, 0.645, 0.487, 0.808, 0.489, 0.525, 0.627, 0.489, 0.67, 0.821, 0.62, 0.618, 0.444, 0.831, 0.668, 0.659, 0.537, 0.63, 0.432, 0.576, 0.688, 0.749, 0.504, 0.584, 0.585, 0.764, 0.555, 0.541, 0.489, 0.709, 0.514, 0.454, 0.779, 0.569, 0.518, 0.561, 0.528, 1.0, 0.552, 0.538, 0.471, 0.641, 0.498, 0.36, 0.489, 0.488, 0.623, 0.591, 0.468, 0.455, 0.472, 0.341, 0.521, 0.498, 0.694, 0.569, 0.485, 0.622, 0.563, 0.603, 0.701, 0.453, 0.432, 0.368, 0.678, 0.519, 0.561], [0.542, 0.563, 0.652, 0.615, 0.615, 0.327, 0.674, 0.665, 0.715, 0.622, 0.496, 0.565, 0.642, 0.514, 0.621, 0.576, 0.533, 0.677, 0.547, 0.574, 0.75, 0.573, 0.549, 0.584, 0.566, 0.54, 0.574, 0.897, 0.65, 0.67, 0.582, 0.603, 0.591, 0.581, 0.668, 0.82, 0.628, 0.782, 0.552, 1.0, 0.7, 0.566, 0.622, 0.682, 0.514, 0.677, 0.615, 0.534, 0.681, 0.615, 0.692, 0.618, 0.41, 0.645, 0.714, 0.636, 0.638, 0.681, 0.528, 0.596, 0.573, 0.6, 0.654, 0.644, 0.455, 0.579, 0.58, 0.615], [0.475, 0.545, 0.595, 0.634, 0.601, 0.382, 0.678, 0.629, 0.618, 0.554, 0.505, 0.466, 0.523, 0.518, 0.606, 0.505, 0.579, 0.725, 0.504, 0.496, 0.615, 0.529, 0.486, 0.52, 0.478, 0.557, 0.572, 0.732, 0.624, 0.562, 0.602, 0.473, 0.513, 0.507, 0.679, 0.726, 0.495, 0.688, 0.538, 0.7, 1.0, 0.649, 0.555, 0.687, 0.491, 0.659, 0.6, 0.543, 0.583, 0.565, 0.665, 0.615, 0.437, 0.558, 0.598, 0.62, 0.502, 0.572, 0.605, 0.533, 0.617, 0.555, 0.59, 0.642, 0.612, 0.524, 0.498, 0.642], [0.467, 0.505, 0.444, 0.645, 0.475, 0.362, 0.646, 0.469, 0.486, 0.455, 0.451, 0.435, 0.455, 0.547, 0.463, 0.451, 0.456, 0.585, 0.443, 0.485, 0.655, 0.405, 0.452, 0.484, 0.391, 0.373, 0.435, 0.563, 0.703, 0.604, 0.521, 0.581, 0.49, 0.551, 0.507, 0.615, 0.592, 0.585, 0.471, 0.566, 0.649, 1.0, 0.457, 0.635, 0.376, 0.533, 0.54, 0.406, 0.622, 0.487, 0.551, 0.519, 0.36, 0.698, 0.568, 0.551, 0.589, 0.745, 0.479, 0.404, 0.462, 0.434, 0.465, 0.552, 0.529, 0.395, 0.55, 0.488], [0.643, 0.652, 0.668, 0.493, 0.692, 0.462, 0.536, 0.757, 0.54, 0.717, 0.575, 0.688, 0.67, 0.401, 0.705, 0.678, 0.626, 0.581, 0.622, 0.466, 0.587, 0.667, 0.645, 0.51, 0.697, 0.634, 0.682, 0.634, 0.528, 0.506, 0.683, 0.534, 0.454, 0.652, 0.602, 0.573, 0.56, 0.63, 0.641, 0.622, 0.555, 0.457, 1.0, 0.573, 0.466, 0.543, 0.56, 0.649, 0.602, 0.538, 0.504, 0.542, 0.432, 0.573, 0.549, 0.659, 0.562, 0.525, 0.609, 0.643, 0.65, 0.711, 0.533, 0.5, 0.355, 0.669, 0.525, 0.496], [0.507, 0.532, 0.553, 0.592, 0.578, 0.359, 0.615, 0.557, 0.647, 0.55, 0.458, 0.531, 0.577, 0.543, 0.566, 0.536, 0.551, 0.659, 0.47, 0.481, 0.615, 0.509, 0.464, 0.517, 0.498, 0.504, 0.54, 0.639, 0.646, 0.545, 0.643, 0.514, 0.542, 0.586, 0.664, 0.631, 0.52, 0.709, 0.498, 0.682, 0.687, 0.635, 0.573, 1.0, 0.476, 0.651, 0.607, 0.503, 0.611, 0.626, 0.686, 0.583, 0.404, 0.628, 0.655, 0.595, 0.527, 0.638, 0.54, 0.512, 0.573, 0.486, 0.605, 0.542, 0.485, 0.495, 0.496, 0.62], [0.486, 0.423, 0.512, 0.425, 0.457, 0.244, 0.475, 0.468, 0.561, 0.519, 0.354, 0.48, 0.496, 0.286, 0.449, 0.483, 0.478, 0.498, 0.455, 0.466, 0.469, 0.476, 0.359, 0.519, 0.502, 0.525, 0.457, 0.517, 0.419, 0.445, 0.414, 0.365, 0.495, 0.364, 0.469, 0.473, 0.372, 0.543, 0.36, 0.514, 0.491, 0.376, 0.466, 0.476, 1.0, 0.556, 0.515, 0.421, 0.45, 0.581, 0.518, 0.539, 0.385, 0.459, 0.447, 0.406, 0.378, 0.447, 0.45, 0.512, 0.521, 0.468, 0.587, 0.558, 0.315, 0.461, 0.342, 0.485], [0.622, 0.566, 0.582, 0.525, 0.559, 0.266, 0.724, 0.538, 0.712, 0.613, 0.479, 0.598, 0.607, 0.391, 0.552, 0.592, 0.561, 0.626, 0.581, 0.502, 0.629, 0.577, 0.486, 0.634, 0.546, 0.55, 0.56, 0.693, 0.547, 0.598, 0.568, 0.489, 0.656, 0.481, 0.566, 0.674, 0.5, 0.696, 0.489, 0.677, 0.659, 0.533, 0.543, 0.651, 0.556, 1.0, 0.751, 0.538, 0.604, 0.731, 0.72, 0.766, 0.386, 0.532, 0.574, 0.487, 0.513, 0.544, 0.568, 0.559, 0.687, 0.567, 0.605, 0.656, 0.409, 0.564, 0.443, 0.673], [0.612, 0.597, 0.568, 0.515, 0.538, 0.279, 0.663, 0.533, 0.62, 0.598, 0.464, 0.606, 0.575, 0.355, 0.547, 0.567, 0.549, 0.659, 0.579, 0.47, 0.598, 0.536, 0.475, 0.607, 0.564, 0.51, 0.544, 0.614, 0.512, 0.575, 0.542, 0.503, 0.627, 0.5, 0.509, 0.645, 0.524, 0.597, 0.488, 0.615, 0.6, 0.54, 0.56, 0.607, 0.515, 0.751, 1.0, 0.523, 0.587, 0.661, 0.664, 0.728, 0.435, 0.545, 0.514, 0.455, 0.537, 0.526, 0.596, 0.535, 0.612, 0.561, 0.567, 0.669, 0.402, 0.531, 0.434, 0.602], [0.566, 0.608, 0.673, 0.383, 0.691, 0.408, 0.458, 0.647, 0.489, 0.673, 0.596, 0.623, 0.633, 0.343, 0.697, 0.688, 0.614, 0.531, 0.581, 0.45, 0.463, 0.703, 0.705, 0.433, 0.628, 0.646, 0.725, 0.56, 0.392, 0.39, 0.648, 0.432, 0.418, 0.529, 0.529, 0.517, 0.46, 0.516, 0.623, 0.534, 0.543, 0.406, 0.649, 0.503, 0.421, 0.538, 0.523, 1.0, 0.47, 0.469, 0.532, 0.493, 0.412, 0.451, 0.442, 0.587, 0.461, 0.411, 0.723, 0.568, 0.727, 0.741, 0.476, 0.522, 0.342, 0.711, 0.413, 0.48], [0.638, 0.565, 0.523, 0.577, 0.57, 0.378, 0.69, 0.56, 0.604, 0.588, 0.519, 0.585, 0.581, 0.514, 0.55, 0.573, 0.516, 0.566, 0.536, 0.533, 0.886, 0.52, 0.594, 0.652, 0.544, 0.496, 0.523, 0.665, 0.698, 0.653, 0.545, 0.729, 0.573, 0.622, 0.558, 0.64, 0.742, 0.692, 0.591, 0.681, 0.583, 0.622, 0.602, 0.611, 0.45, 0.604, 0.587, 0.47, 1.0, 0.591, 0.57, 0.581, 0.412, 0.77, 0.598, 0.574, 0.756, 0.734, 0.455, 0.479, 0.531, 0.535, 0.545, 0.511, 0.426, 0.51, 0.577, 0.576], [0.591, 0.519, 0.545, 0.545, 0.515, 0.258, 0.616, 0.521, 0.691, 0.584, 0.457, 0.618, 0.615, 0.371, 0.513, 0.587, 0.562, 0.569, 0.527, 0.476, 0.617, 0.541, 0.448, 0.66, 0.578, 0.532, 0.515, 0.613, 0.528, 0.571, 0.49, 0.478, 0.605, 0.47, 0.542, 0.566, 0.489, 0.638, 0.468, 0.615, 0.565, 0.487, 0.538, 0.626, 0.581, 0.731, 0.661, 0.469, 0.591, 1.0, 0.668, 0.722, 0.389, 0.559, 0.517, 0.47, 0.496, 0.566, 0.481, 0.554, 0.547, 0.527, 0.695, 0.608, 0.359, 0.524, 0.407, 0.653], [0.533, 0.579, 0.577, 0.504, 0.536, 0.278, 0.64, 0.531, 0.73, 0.554, 0.46, 0.581, 0.658, 0.402, 0.529, 0.569, 0.53, 0.636, 0.49, 0.493, 0.604, 0.547, 0.478, 0.584, 0.534, 0.558, 0.537, 0.66, 0.555, 0.521, 0.547, 0.481, 0.575, 0.489, 0.573, 0.688, 0.481, 0.64, 0.455, 0.692, 0.665, 0.551, 0.504, 0.686, 0.518, 0.72, 0.664, 0.532, 0.57, 0.668, 1.0, 0.687, 0.42, 0.538, 0.551, 0.52, 0.488, 0.566, 0.581, 0.588, 0.62, 0.537, 0.616, 0.654, 0.462, 0.541, 0.432, 0.632], [0.575, 0.559, 0.524, 0.486, 0.524, 0.25, 0.67, 0.507, 0.686, 0.587, 0.461, 0.596, 0.574, 0.354, 0.528, 0.578, 0.541, 0.631, 0.531, 0.443, 0.621, 0.545, 0.468, 0.624, 0.538, 0.51, 0.535, 0.617, 0.506, 0.547, 0.514, 0.476, 0.601, 0.469, 0.51, 0.651, 0.487, 0.634, 0.472, 0.618, 0.615, 0.519, 0.542, 0.583, 0.539, 0.766, 0.728, 0.493, 0.581, 0.722, 0.687, 1.0, 0.394, 0.504, 0.504, 0.442, 0.494, 0.515, 0.519, 0.571, 0.556, 0.542, 0.581, 0.7, 0.431, 0.539, 0.395, 0.695], [0.369, 0.48, 0.398, 0.335, 0.384, 0.229, 0.395, 0.417, 0.384, 0.388, 0.299, 0.398, 0.391, 0.233, 0.375, 0.369, 0.402, 0.494, 0.367, 0.313, 0.387, 0.348, 0.355, 0.343, 0.434, 0.392, 0.352, 0.403, 0.344, 0.302, 0.397, 0.326, 0.29, 0.328, 0.331, 0.502, 0.337, 0.442, 0.341, 0.41, 0.437, 0.36, 0.432, 0.404, 0.385, 0.386, 0.435, 0.412, 0.412, 0.389, 0.42, 0.394, 1.0, 0.396, 0.325, 0.403, 0.347, 0.371, 0.405, 0.306, 0.429, 0.365, 0.483, 0.436, 0.279, 0.345, 0.29, 0.335], [0.578, 0.518, 0.525, 0.631, 0.548, 0.414, 0.662, 0.556, 0.548, 0.584, 0.455, 0.563, 0.528, 0.594, 0.529, 0.55, 0.506, 0.588, 0.538, 0.628, 0.714, 0.481, 0.559, 0.615, 0.492, 0.449, 0.496, 0.63, 0.727, 0.69, 0.53, 0.672, 0.568, 0.593, 0.547, 0.607, 0.685, 0.687, 0.521, 0.645, 0.558, 0.698, 0.573, 0.628, 0.459, 0.532, 0.545, 0.451, 0.77, 0.559, 0.538, 0.504, 0.396, 1.0, 0.607, 0.577, 0.69, 0.897, 0.42, 0.462, 0.496, 0.502, 0.526, 0.539, 0.441, 0.483, 0.625, 0.496], [0.471, 0.525, 0.522, 0.562, 0.602, 0.329, 0.554, 0.549, 0.625, 0.564, 0.469, 0.525, 0.546, 0.573, 0.585, 0.524, 0.479, 0.65, 0.471, 0.562, 0.602, 0.509, 0.456, 0.504, 0.509, 0.45, 0.543, 0.672, 0.624, 0.548, 0.572, 0.591, 0.534, 0.566, 0.63, 0.624, 0.579, 0.654, 0.498, 0.714, 0.598, 0.568, 0.549, 0.655, 0.447, 0.574, 0.514, 0.442, 0.598, 0.517, 0.551, 0.504, 0.325, 0.607, 1.0, 0.55, 0.587, 0.616, 0.462, 0.5, 0.476, 0.507, 0.51, 0.511, 0.438, 0.557, 0.496, 0.519], [0.613, 0.598, 0.737, 0.603, 0.698, 0.504, 0.555, 0.69, 0.541, 0.634, 0.65, 0.601, 0.675, 0.52, 0.698, 0.607, 0.66, 0.561, 0.61, 0.479, 0.579, 0.638, 0.654, 0.484, 0.563, 0.602, 0.668, 0.622, 0.642, 0.529, 0.669, 0.518, 0.466, 0.679, 0.64, 0.583, 0.534, 0.591, 0.694, 0.636, 0.62, 0.551, 0.659, 0.595, 0.406, 0.487, 0.455, 0.587, 0.574, 0.47, 0.52, 0.442, 0.403, 0.577, 0.55, 1.0, 0.542, 0.584, 0.584, 0.534, 0.627, 0.645, 0.478, 0.481, 0.451, 0.617, 0.583, 0.569], [0.59, 0.548, 0.536, 0.529, 0.567, 0.399, 0.594, 0.575, 0.577, 0.561, 0.515, 0.57, 0.571, 0.461, 0.544, 0.575, 0.476, 0.491, 0.553, 0.521, 0.729, 0.527, 0.583, 0.556, 0.511, 0.434, 0.524, 0.604, 0.63, 0.626, 0.575, 0.89, 0.543, 0.62, 0.55, 0.585, 0.973, 0.6, 0.569, 0.638, 0.502, 0.589, 0.562, 0.527, 0.378, 0.513, 0.537, 0.461, 0.756, 0.496, 0.488, 0.494, 0.347, 0.69, 0.587, 0.542, 1.0, 0.665, 0.433, 0.514, 0.466, 0.552, 0.47, 0.468, 0.37, 0.512, 0.599, 0.504], [0.538, 0.489, 0.53, 0.667, 0.506, 0.351, 0.683, 0.543, 0.578, 0.535, 0.446, 0.515, 0.533, 0.613, 0.492, 0.532, 0.462, 0.569, 0.5, 0.598, 0.758, 0.455, 0.516, 0.577, 0.455, 0.426, 0.46, 0.665, 0.773, 0.734, 0.491, 0.64, 0.6, 0.582, 0.547, 0.639, 0.654, 0.711, 0.485, 0.681, 0.572, 0.745, 0.525, 0.638, 0.447, 0.544, 0.526, 0.411, 0.734, 0.566, 0.566, 0.515, 0.371, 0.897, 0.616, 0.584, 0.665, 1.0, 0.404, 0.465, 0.457, 0.482, 0.554, 0.567, 0.465, 0.464, 0.636, 0.507], [0.562, 0.623, 0.668, 0.42, 0.68, 0.376, 0.475, 0.68, 0.522, 0.614, 0.588, 0.582, 0.633, 0.357, 0.707, 0.599, 0.637, 0.595, 0.56, 0.434, 0.47, 0.635, 0.579, 0.45, 0.634, 0.619, 0.683, 0.567, 0.421, 0.387, 0.637, 0.381, 0.401, 0.566, 0.552, 0.563, 0.428, 0.51, 0.622, 0.528, 0.605, 0.479, 0.609, 0.54, 0.45, 0.568, 0.596, 0.723, 0.455, 0.481, 0.581, 0.519, 0.405, 0.42, 0.462, 0.584, 0.433, 0.404, 1.0, 0.582, 0.762, 0.68, 0.49, 0.557, 0.367, 0.638, 0.432, 0.512], [0.566, 0.546, 0.662, 0.433, 0.61, 0.411, 0.464, 0.64, 0.615, 0.629, 0.582, 0.62, 0.651, 0.387, 0.642, 0.653, 0.582, 0.526, 0.603, 0.469, 0.528, 0.69, 0.573, 0.507, 0.632, 0.668, 0.672, 0.594, 0.464, 0.47, 0.593, 0.483, 0.513, 0.529, 0.621, 0.529, 0.506, 0.548, 0.563, 0.596, 0.533, 0.404, 0.643, 0.512, 0.512, 0.559, 0.535, 0.568, 0.479, 0.554, 0.588, 0.571, 0.306, 0.462, 0.5, 0.534, 0.514, 0.465, 0.582, 1.0, 0.6, 0.697, 0.507, 0.534, 0.368, 0.678, 0.46, 0.58], [0.662, 0.676, 0.736, 0.438, 0.676, 0.402, 0.531, 0.688, 0.583, 0.676, 0.58, 0.653, 0.7, 0.372, 0.669, 0.68, 0.685, 0.593, 0.679, 0.473, 0.504, 0.703, 0.613, 0.528, 0.68, 0.721, 0.699, 0.577, 0.445, 0.438, 0.673, 0.434, 0.465, 0.53, 0.554, 0.566, 0.455, 0.573, 0.603, 0.573, 0.617, 0.462, 0.65, 0.573, 0.521, 0.687, 0.612, 0.727, 0.531, 0.547, 0.62, 0.556, 0.429, 0.496, 0.476, 0.627, 0.466, 0.457, 0.762, 0.6, 1.0, 0.695, 0.535, 0.573, 0.35, 0.684, 0.476, 0.519], [0.628, 0.639, 0.781, 0.436, 0.786, 0.432, 0.491, 0.724, 0.578, 0.765, 0.749, 0.7, 0.722, 0.379, 0.794, 0.828, 0.63, 0.559, 0.702, 0.48, 0.545, 0.894, 0.789, 0.478, 0.658, 0.746, 0.873, 0.645, 0.487, 0.492, 0.676, 0.502, 0.502, 0.598, 0.563, 0.562, 0.55, 0.563, 0.701, 0.6, 0.555, 0.434, 0.711, 0.486, 0.468, 0.567, 0.561, 0.741, 0.535, 0.527, 0.537, 0.542, 0.365, 0.502, 0.507, 0.645, 0.552, 0.482, 0.68, 0.697, 0.695, 1.0, 0.527, 0.533, 0.337, 0.881, 0.506, 0.548], [0.496, 0.486, 0.56, 0.498, 0.502, 0.225, 0.559, 0.558, 0.663, 0.532, 0.431, 0.5, 0.568, 0.352, 0.506, 0.516, 0.5, 0.607, 0.454, 0.485, 0.602, 0.486, 0.437, 0.512, 0.542, 0.498, 0.484, 0.653, 0.55, 0.529, 0.451, 0.44, 0.512, 0.466, 0.492, 0.598, 0.457, 0.673, 0.453, 0.654, 0.59, 0.465, 0.533, 0.605, 0.587, 0.605, 0.567, 0.476, 0.545, 0.695, 0.616, 0.581, 0.483, 0.526, 0.51, 0.478, 0.47, 0.554, 0.49, 0.507, 0.535, 0.527, 1.0, 0.615, 0.314, 0.494, 0.416, 0.538], [0.516, 0.535, 0.564, 0.559, 0.515, 0.265, 0.653, 0.533, 0.599, 0.622, 0.412, 0.513, 0.523, 0.412, 0.519, 0.5, 0.508, 0.664, 0.558, 0.537, 0.561, 0.5, 0.453, 0.541, 0.506, 0.475, 0.498, 0.651, 0.556, 0.616, 0.485, 0.445, 0.629, 0.445, 0.5, 0.686, 0.456, 0.624, 0.432, 0.644, 0.642, 0.552, 0.5, 0.542, 0.558, 0.656, 0.669, 0.522, 0.511, 0.608, 0.654, 0.7, 0.436, 0.539, 0.511, 0.481, 0.468, 0.567, 0.557, 0.534, 0.573, 0.533, 0.615, 1.0, 0.445, 0.507, 0.442, 0.594], [0.34, 0.352, 0.387, 0.527, 0.398, 0.43, 0.498, 0.379, 0.37, 0.383, 0.347, 0.327, 0.337, 0.545, 0.406, 0.332, 0.425, 0.468, 0.365, 0.418, 0.437, 0.335, 0.352, 0.397, 0.267, 0.355, 0.381, 0.456, 0.478, 0.477, 0.417, 0.363, 0.418, 0.368, 0.478, 0.482, 0.364, 0.444, 0.368, 0.455, 0.612, 0.529, 0.355, 0.485, 0.315, 0.409, 0.402, 0.342, 0.426, 0.359, 0.462, 0.431, 0.279, 0.441, 0.438, 0.451, 0.37, 0.465, 0.367, 0.368, 0.35, 0.337, 0.314, 0.445, 1.0, 0.337, 0.424, 0.473], [0.625, 0.598, 0.746, 0.4, 0.773, 0.416, 0.473, 0.655, 0.562, 0.76, 0.731, 0.696, 0.718, 0.389, 0.789, 0.844, 0.639, 0.523, 0.672, 0.45, 0.513, 0.878, 0.763, 0.477, 0.648, 0.733, 0.89, 0.589, 0.452, 0.451, 0.678, 0.465, 0.494, 0.568, 0.566, 0.531, 0.5, 0.542, 0.678, 0.579, 0.524, 0.395, 0.669, 0.495, 0.461, 0.564, 0.531, 0.711, 0.51, 0.524, 0.541, 0.539, 0.345, 0.483, 0.557, 0.617, 0.512, 0.464, 0.638, 0.678, 0.684, 0.881, 0.494, 0.507, 0.337, 1.0, 0.453, 0.545], [0.519, 0.474, 0.53, 0.55, 0.5, 0.485, 0.541, 0.574, 0.465, 0.489, 0.463, 0.465, 0.479, 0.569, 0.492, 0.485, 0.446, 0.474, 0.55, 0.588, 0.604, 0.472, 0.522, 0.462, 0.417, 0.458, 0.45, 0.564, 0.614, 0.641, 0.507, 0.57, 0.502, 0.556, 0.533, 0.545, 0.596, 0.566, 0.519, 0.58, 0.498, 0.55, 0.525, 0.496, 0.342, 0.443, 0.434, 0.413, 0.577, 0.407, 0.432, 0.395, 0.29, 0.625, 0.496, 0.583, 0.599, 0.636, 0.432, 0.46, 0.476, 0.506, 0.416, 0.442, 0.424, 0.453, 1.0, 0.417], [0.577, 0.469, 0.569, 0.571, 0.565, 0.328, 0.596, 0.518, 0.67, 0.577, 0.561, 0.544, 0.538, 0.458, 0.595, 0.519, 0.614, 0.594, 0.575, 0.457, 0.618, 0.545, 0.51, 0.646, 0.479, 0.522, 0.598, 0.637, 0.59, 0.547, 0.534, 0.479, 0.637, 0.532, 0.627, 0.592, 0.49, 0.64, 0.561, 0.615, 0.642, 0.488, 0.496, 0.62, 0.485, 0.673, 0.602, 0.48, 0.576, 0.653, 0.632, 0.695, 0.335, 0.496, 0.519, 0.569, 0.504, 0.507, 0.512, 0.58, 0.519, 0.548, 0.538, 0.594, 0.473, 0.545, 0.417, 1.0]]