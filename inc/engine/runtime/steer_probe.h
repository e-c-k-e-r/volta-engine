#pragma once

#include <vector>

namespace engine::runtime {

// Host maths of the steering probe (see Generator::probe_slot): directions between two sets of residual states at one layer, and the gate score.
// A state set is a list of pointers to `hidden` floats.
using StateSet = std::vector<const float*>;

struct ProbeAxis {
	std::vector<float> dir;   // unit direction from set a to set b
	float pa = 0.0f;          // mean projection of a on dir
	float pb = 0.0f;          // mean projection of b on dir
	float sd = 0.0f;          // standard deviation of a's projections
};

// dir = normalize( mean( b ) - mean( a ) ); with `orthogonalize`, first stripped of what varies inside a (the deviations of a's samples from their
// mean, projected out one after the other), so the axis separates the sets and not the spread of the first one
ProbeAxis probe_axis( const StateSet& a, const StateSet& b, int hidden, bool orthogonalize );

// More refusal directions: `count` further axes from the residual variation of set b around a's mean once `used` (the directions already taken,
// unit vectors) are removed. The b samples' deviations from a's mean are stripped of the used directions, the strongest remaining direction is found by
// power iteration (the top singular vector of the deviations), stripped of a's own spread as probe_axis does, and taken; the next round starts
// from the deviations without it. Each axis has its own pa / pb / sd along it. Stops early when what is left of the deviations is negligible.
std::vector<ProbeAxis> probe_extra_axes( const StateSet& a, const StateSet& b, int hidden, const std::vector<std::vector<float>>& used, int count );

// How far a state lies from set a towards set b along the axis: 0 at a's mean, 1 at b's (the gap floored so a degenerate axis does not blow up)
float probe_fraction( const ProbeAxis& axis, const float* state, int hidden );

} // namespace engine::runtime
