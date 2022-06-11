
//! A worker pool used for parallel computing of a large number of relatively small tasks.
//!
//! Tasks are computed on separate threads and are given read-only access to a common context.
//!
//! Threads are spawned at creation and do not currently recover from panics.

use std::sync::Arc;
use std::{thread};
use std::thread::JoinHandle;
use crossbeam::channel::{TryRecvError, unbounded};


/// Abstraction of a worker that can execute computations
///
/// Although a shared context is provided for each computation, a Worker itself can contain state which cannot be shared across threads.
///
/// The execute function should be non blocking as a blocking execute function can lead to deadlocks when dropping the WorkersPool.
/// The execute function should be non panicking as a panic leads to the thread executing the worker shutting down.
pub trait Worker: Default  {
    type Data: 'static + Send;
    type Result: 'static + Send;
    type Context: 'static + Send + Sync;

    fn execute(&mut self, data: Self::Data, context: &Arc<Self::Context>) -> Self::Result;
}

/// Abstraction of a threadpool for executing units of computation in parallel.
pub struct WorkersPool<W: Worker> {
    result_receiver: crossbeam::channel::Receiver<W::Result>,
    work_sender: crossbeam::channel::Sender<W::Data>,
    #[allow(dead_code)]
    workers: Vec<JoinHandle<()>>
}

impl<W: Worker> WorkersPool<W> {
    pub fn new(context: W::Context) -> Self {
        let (result_sender,result_receiver) = unbounded();
        let (work_sender,work_receiver) = unbounded();

        let context = Arc::new(context);

        let thread_count = num_cpus::get();

        let mut workers = vec![];

        for _ in 0..thread_count {
            let work_receiver = work_receiver.clone();
            let result_sender = result_sender.clone();

            let context_clone = context.clone();

            let thread = thread::spawn(move || {
                let mut worker = W::default();
                let context = context_clone;

                loop {
                    let work = work_receiver.recv();

                    let work = match work {
                        Err(_) => {
                            return;
                        },
                        Ok(work) => {
                            work
                        }
                    };

                    let result = worker.execute(work, &context);

                    let send_result = result_sender.send(result);

                    match send_result {
                        Ok(_) => {}
                        Err(_) => {
                            return;
                        }
                    }
                }
            });

            workers.push(thread);
        }

        Self {
            result_receiver,
            work_sender,
            workers
        }
    }

    /// Adds work to be executed on one of the threads of this pool
    /// This function is non-blocking
    pub fn add_work(&mut self, work: W::Data) -> Result<(),()>{
        self.work_sender.send(work)
            .map_err(|_| ())?;

        Ok(())
    }

    /// Receives the result of a computation
    /// This function blocks until a result is available or all threads have panicked
    pub fn receive_result(&mut self) -> Result<W::Result, ()> {
        self.result_receiver.recv().map_err(|_| ())
    }

    /// Tries to receive the result of a computation
    /// This function does not block if no result is available but rather returns Ok(None)
    pub fn try_receive_result(&mut self) -> Result<Option<W::Result>, ()> {
        let result = self.result_receiver.try_recv();

        match result {
            Err(err) => {
                match err {
                    TryRecvError::Empty => {
                        Ok(None)
                    }
                    TryRecvError::Disconnected => {
                        Err(())
                    }
                }
            }
            Ok(ok) => {
                Ok(Some(ok))
            }
        }
    }

    /// Collects all available results
    pub fn collect_finished(&mut self) -> Result<Vec<W::Result>, ()> {
        let mut results = vec![];

        loop {
            let result = self.try_receive_result()?;
            match result {
                None => break,
                Some(result) => {
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Checks if any work is left in the work queue
    ///
    /// Note: An empty work queue does not mean that none of the worker threads is still busy
    pub fn has_work_left(&self) -> bool {
        self.work_sender.is_empty()
    }

    /// Checks if there are results available
    pub fn has_results(&self) -> bool {
        !self.result_receiver.is_empty()
    }
}
