#pragma once

#include <cstdint>

// The mailbox a rank's device and the CPU expert service share (pinned, mapped host memory). Per MoE layer the device posts the routes it hands
// to the CPU (and their input rows), the service computes them and publishes the summed expert outputs.
namespace engine::cpu_moe {

constexpr int kMailboxRows = 16;                       // most rows one request carries (pod::MAX_BATCH)
constexpr int kMailboxRoutes = kMailboxRows * 16;      // most routes: rows x top_k

struct MailboxRoute {
	int32_t row;
	int32_t expert;       // global expert id
	float weight;
};

struct Mailbox {
	alignas( 64 ) volatile uint64_t req_seq;   // written by the device last (after the payload), read by the service
	alignas( 64 ) volatile uint64_t done_seq;  // written by the service last (after the result), read by the device
	alignas( 64 ) int32_t layer, rows, nroutes, hidden;
	volatile int32_t err;                      // set by the device when the service did not answer in time
	volatile uint64_t wait_ns, waits;          // device statistics: time the wait kernels spent waiting for the service, and how many ran
	MailboxRoute routes[kMailboxRoutes];
};

// bytes of a rank's mailbox block: the header, then x (FP16 [kMailboxRows][hidden]), then out (FP32 [kMailboxRows][hidden])
inline uint64_t mailbox_bytes( int hidden ) { return ( ( sizeof( Mailbox ) + 255 ) & ~uint64_t( 255 ) ) + uint64_t( kMailboxRows ) * hidden * ( 2 + 4 ); }
inline uint16_t* mailbox_x( Mailbox* mb ) { return reinterpret_cast<uint16_t*>( reinterpret_cast<char*>( mb ) + ( ( sizeof( Mailbox ) + 255 ) & ~uint64_t( 255 ) ) ); }
inline float* mailbox_out( Mailbox* mb, int hidden ) { return reinterpret_cast<float*>( mailbox_x( mb ) + uint64_t( kMailboxRows ) * hidden ); }

}   // namespace engine::cpu_moe
