#pragma once

#include "engine/video/h3_pipeline.h"
#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::server {

double wall_seconds();

// One generation job. Fields other than `cancel` are guarded by JobStore's mutex.
struct Job {
	int id = 0;
	std::string state = "queued"; // queued | running | done | failed | cancelled
	std::string stage, error, dir;
	nlohmann::json params; // what the client asked for (kept in the history)
	int step = 0, steps = 0;
	double progress = 0, created = 0, started = 0, finished = 0, t_text = 0, t_load = 0, t_denoise = 0, t_decode = 0, t_total = 0;
	std::atomic<bool> cancel{ false };

	// inputs of the run (not persisted): adapters, uploaded files
	std::vector<engine::video::H3Lora> loras;
	std::vector<std::string> images, ref_images, ref_videos, ref_audios;
	std::vector<int> frame_indices;
	int steps_total = 0;
};

using JobPtr = std::shared_ptr<Job>;

// The queue and the history of jobs (<out_dir>/jobs/<id>/meta.json). One worker takes jobs with next() and reports with update() / finish().
class JobStore {
public:
	// Creates the directory and loads the history; jobs that were queued or running when the server stopped come back as failed.
	void init( const std::string& out_dir );

	JobPtr create();                  // assigns an id and the job directory
	void enqueue( const JobPtr& j );  // saves the metadata and wakes the worker
	JobPtr find( int id ) const;
	nlohmann::json describe( const JobPtr& j ) const;
	nlohmann::json list() const;      // newest first
	void cancel( const JobPtr& j );
	bool remove( const JobPtr& j, std::string* error );

	// Worker side: blocks until a job is queued (nullptr after stop()); marks it running. With idle_s > 0 it also returns nullptr (and *idle = true) when nothing
	// was queued for that long.
	// poke() makes it return nullptr with *poked = true once (ahead of queued jobs): the worker has other work (caption requests).
	JobPtr next( double idle_s = 0, bool* idle = nullptr, bool* poked = nullptr );
	void poke();
	template <typename F>
	void update( const JobPtr& j, F fn ) {
		std::lock_guard<std::mutex> lk( mu_ );
		fn( *j );
	}
	// Marks the end of a run (done / failed / cancelled) and saves the metadata.
	void finish( const JobPtr& j, bool ok, bool cancelled, const engine::video::H3GenResult& res, double started_at );
	void stop();

	struct Counts {
		int queued = 0, running = 0, done = 0, failed = 0, cancelled = 0;
		double last_total_s = 0, last_denoise_s = 0;
	};
	Counts counts() const;

private:
	nlohmann::json describe_locked( const Job& j ) const;
	void save_locked( const Job& j ) const;

	std::string out_dir_;
	mutable std::mutex mu_;
	std::condition_variable cv_;
	std::map<int, JobPtr> jobs_;
	std::deque<int> queue_;
	int next_id_ = 1;
	bool stopping_ = false;
	bool poked_ = false;
};

} // namespace engine::server
