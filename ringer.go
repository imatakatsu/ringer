// Package ringer provides high-performance, lock-free data structures.
// It includes an MPMC (multi-producer multi-consumer) RingBuffer 
// and a thread-safe Rotator for load balancing.
package ringer

import (
	"errors"
	"iter"
	"math/bits"
	"runtime"
	"sync/atomic"
)

var (
	// ErrBufferFull is returned by Push when the buffer has no available capacity.
	ErrBufferFull = errors.New("buffer is full")
	// ErrBufferEmpty is returned by Pop when the buffer has no items to retrieve.
	ErrBufferEmpty = errors.New("buffer is empty")
)

// RingBuffer is a lock-free MPMC (multi-producer multi-consumer) concurrent queue.
// It uses Vyukov's algorithm and is optimized for low latency with padding 
// to prevent false sharing between CPU cache lines.
type RingBuffer[T any] struct {
	_    [64]byte      // padding for false sharing
	head atomic.Uint64 // consumer index
	_    [64]byte      // padding
	tail atomic.Uint64 // producer index
	_    [64]byte      // padding
	mask uint64        // size - 1, for power-of-two masking
	buf  []slot[T]     // actual storage
}

// slot represents a single storage unit in the RingBuffer.
type slot[T any] struct {
	step atomic.Uint64
	ptr  atomic.Pointer[T]
}

// NewBuffer creates a new RingBuffer of the given size.
// The size is automatically rounded up to the nearest power of two.
func NewBuffer[T any](size uint64) *RingBuffer[T] {
	if size < 2 {
		size = 2
	}
	size = 1 << bits.Len64(size-1)

	rb := &RingBuffer[T]{
		mask: size - 1,
		buf:  make([]slot[T], size),
	}

	for i := uint64(0); i < size; i++ {
		rb.buf[i].step.Store(i)
	}
	return rb
}

// BufferFromSlice creates a pre-populated RingBuffer from an existing slice.
// The internal capacity is rounded up to the nearest power of two.
func BufferFromSlice[T any](slice []T) *RingBuffer[T] {
	origLen := uint64(len(slice))
	size := max(origLen, 2)
	size = 1 << bits.Len64(size-1)

	buf := make([]slot[T], size)

	for i := uint64(0); i < size; i++ {
		if i < origLen {
			buf[i].step.Store(i + 1)
			val := slice[i]
			buf[i].ptr.Store(&val)
		} else {
			buf[i].step.Store(i)
		}
	}

	rbuf := &RingBuffer[T]{
		mask: size - 1,
		buf:  buf,
	}
	rbuf.tail.Store(origLen)
	return rbuf
}

// Push adds an item to the buffer.
// Returns ErrBufferFull if the buffer is full.
// This operation is lock-free and thread-safe.
func (b *RingBuffer[T]) Push(item *T) error {
	for {
		t := b.tail.Load()
		s := &b.buf[t&b.mask]
		step := s.step.Load()

		if step == t {
			if b.tail.CompareAndSwap(t, t+1) {
				s.ptr.Store(item)
				s.step.Store(t + 1)
				return nil
			}
		} else if step < t {
			return ErrBufferFull
		}

		runtime.Gosched()
	}
}

// Pop removes and returns an item from the buffer.
// Returns ErrBufferEmpty if the buffer is empty.
// This operation is lock-free and thread-safe.
func (b *RingBuffer[T]) Pop() (*T, error) {
	for {
		h := b.head.Load()
		s := &b.buf[h&b.mask]
		step := s.step.Load()

		if step == h+1 {
			if b.head.CompareAndSwap(h, h+1) {
				res := s.ptr.Swap(nil)
				s.step.Store(h + uint64(len(b.buf)))
				return res, nil
			}
		} else if step < h+1 {
			return nil, ErrBufferEmpty
		}

		runtime.Gosched()
	}
}

// Rotator is a thread-safe round-robin load balancer for a static slice of items.
type Rotator[T any] struct {
	idx   atomic.Uint64
	items []T
}

// NewRotator creates a Rotator from a slice.
// If the slice is empty, it populates it with a single zero-value item to prevent panics.
func NewRotator[T any](slice []T) *Rotator[T] {
	if len(slice) == 0 {
		var zero T
		slice = append(slice, zero)
	}
	return &Rotator[T]{
		items: slice,
	}
}

// Next returns the next item in the rotation sequence.
// It is thread-safe and uses atomic increments to ensure fair distribution.
func (r *Rotator[T]) Next() T {
	idx := r.idx.Add(1) - 1
	return r.items[idx%uint64(len(r.items))]
}

// Ring returns an infinite iterator (Go 1.23+) that cycles through items indefinitely.
func (r *Rotator[T]) Ring() iter.Seq[T] {
	return func(yield func(T) bool) {
		for {
			if !yield(r.Next()) {
				return
			}
		}
	}
}

// Iter returns a finite iterator (Go 1.23+) that yields each item once along with its index.
func (r *Rotator[T]) Iter() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, e := range r.items {
			if !yield(i, e) {
				return
			}
		}
	}
}