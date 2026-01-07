package crypto

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"

	"github.com/cloudflare/circl/oprf"
	"github.com/cloudflare/circl/zk/dleq"
)

// OPRF Protocol Abstraction
//
// This module currently uses Ristretto255-based OPRF from cloudflare/circl.
// Ristretto255 provides ~128-bit classical security but is NOT post-quantum secure.
//
// POST-QUANTUM MIGRATION PATH:
//
// When NIST standardizes lattice-based OPRF (expected 2026-2027), replace this
// implementation with a lattice-based variant. The interface is designed to make
// this swap transparent to callers.
//
// Options for post-quantum OPRF:
// 1. Lattice-based OPRF using Module-LWE (research stage)
// 2. Isogeny-based OPRF using CSIDH (experimental)
// 3. Hash-based approach with Module-SIS commitments
//
// For now, the system provides:
// - ML-DSA (Dilithium) signatures: Post-quantum secure
// - OPRF: Classical security only (Ristretto255)
// - Split Accumulator: Based on Module-SIS (post-quantum)
//
// The OPRF layer is the weakest link. A quantum adversary could potentially
// derive the OPRF key and perform offline dictionary attacks on hashes.
// Mitigation: Rotate OPRF keys frequently and treat matches as probabilistic.

// OPRFSuite identifies the OPRF cipher suite in use.
type OPRFSuite string

const (
	// SuiteRistretto255 uses the Ristretto255 group (classical security only)
	SuiteRistretto255 OPRFSuite = "ristretto255"
	// SuiteLatticePQ is reserved for future post-quantum implementation
	SuiteLatticePQ OPRFSuite = "lattice-pq"
)

// CurrentOPRFSuite identifies which suite is currently active.
// This allows clients and servers to negotiate or verify compatibility.
const CurrentOPRFSuite = SuiteRistretto255

var oprfSuite = oprf.SuiteRistretto255

// OPRFClientInterface defines the client-side OPRF operations.
// Implementations can swap between classical and post-quantum variants.
type OPRFClientInterface interface {
	// Blind creates a blinded input and returns state needed for finalization.
	Blind(input []byte) (state interface{}, blindedRequest []byte, err error)
	// Finalize unblinds the server's response to get the OPRF output.
	Finalize(state interface{}, evaluation []byte) ([]byte, error)
	// Suite returns the cipher suite identifier.
	Suite() OPRFSuite
}

// OPRFServerInterface defines the server-side OPRF operations.
type OPRFServerInterface interface {
	// Evaluate computes the OPRF on a blinded input.
	Evaluate(blindedRequest []byte) ([]byte, error)
	// Suite returns the cipher suite identifier.
	Suite() OPRFSuite
}

// VOPRFClient implements verifiable OPRF client using Ristretto255.
// VOPRF mode allows clients to verify that the server used the correct key,
// addressing the "opaque list" critique where clients cannot audit server behavior.
type VOPRFClient struct {
	client    oprf.VerifiableClient
	publicKey *oprf.PublicKey
}

// OPRFClient wraps VOPRFClient for backward compatibility.
// Now uses verifiable mode by default for auditability.
type OPRFClient struct {
	voprf *VOPRFClient
}

// OPRFServer implements OPRFServerInterface using Ristretto255 in verifiable mode.
// The mutex protects against a race in CIRCL's lazy public key initialization.
type OPRFServer struct {
	server oprf.VerifiableServer
	mu     sync.Mutex // Protects concurrent Evaluate calls (CIRCL race workaround)
}

// OPRFState holds the blinding state needed for finalization.
type OPRFState struct {
	finalize  *oprf.FinalizeData
	publicKey *oprf.PublicKey // Server's public key for verification
}

// NewVOPRFClient creates a new verifiable OPRF client with the server's public key.
// The public key enables clients to verify the server is using the correct OPRF key.
func NewVOPRFClient(serverPublicKey *oprf.PublicKey) *VOPRFClient {
	return &VOPRFClient{
		client:    oprf.NewVerifiableClient(oprfSuite, serverPublicKey),
		publicKey: serverPublicKey,
	}
}

// NewOPRFClient creates a new OPRF client using Ristretto255.
// Note: For full verifiability, use NewOPRFClientWithPublicKey instead.
func NewOPRFClient() *OPRFClient {
	// Create with nil public key - will be set when SetServerPublicKey is called
	return &OPRFClient{voprf: nil}
}

// NewOPRFClientWithPublicKey creates a verifiable OPRF client with server's public key.
func NewOPRFClientWithPublicKey(serverPublicKey *oprf.PublicKey) *OPRFClient {
	return &OPRFClient{voprf: NewVOPRFClient(serverPublicKey)}
}

// SetServerPublicKey configures the server's public key for verification.
func (c *OPRFClient) SetServerPublicKey(publicKey *oprf.PublicKey) {
	c.voprf = NewVOPRFClient(publicKey)
}

// NewOPRFServer creates a new OPRF server with the given private key in verifiable mode.
func NewOPRFServer(privateKey *oprf.PrivateKey) *OPRFServer {
	return &OPRFServer{server: oprf.NewVerifiableServer(oprfSuite, privateKey)}
}

// Suite returns the cipher suite identifier.
func (c *OPRFClient) Suite() OPRFSuite {
	return SuiteRistretto255
}

// Suite returns the cipher suite identifier.
func (s *OPRFServer) Suite() OPRFSuite {
	return SuiteRistretto255
}

// GenerateOPRFKeyPair generates a new OPRF key pair.
func GenerateOPRFKeyPair() (privateKey []byte, publicKey []byte, err error) {
	key, err := oprf.GenerateKey(oprfSuite, rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	privateKey, err = key.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}
	publicKey, err = key.Public().MarshalBinary()
	if err != nil {
		return nil, nil, err
	}
	return privateKey, publicKey, nil
}

// ParseOPRFPrivateKey deserializes an OPRF private key.
func ParseOPRFPrivateKey(data []byte) (*oprf.PrivateKey, error) {
	key := &oprf.PrivateKey{}
	if err := key.UnmarshalBinary(oprfSuite, data); err != nil {
		return nil, err
	}
	return key, nil
}

// ParseOPRFPublicKey deserializes an OPRF public key.
func ParseOPRFPublicKey(data []byte) (*oprf.PublicKey, error) {
	key := &oprf.PublicKey{}
	if err := key.UnmarshalBinary(oprfSuite, data); err != nil {
		return nil, err
	}
	return key, nil
}

// Blind creates a blinded OPRF request.
func (c *OPRFClient) Blind(input []byte) (*OPRFState, []byte, error) {
	if c == nil || c.voprf == nil {
		return nil, nil, errors.New("oprf client is nil or not initialized with public key")
	}
	fin, req, err := c.voprf.client.Blind([][]byte{input})
	if err != nil {
		return nil, nil, err
	}
	wire, err := marshalOPRFElements(req.Elements)
	if err != nil {
		return nil, nil, err
	}
	return &OPRFState{finalize: fin, publicKey: c.voprf.publicKey}, wire, nil
}

// Finalize unblinds the server response and verifies the OPRF proof.
// In verifiable mode, this also checks that the server used the correct key.
func (c *OPRFClient) Finalize(state *OPRFState, evalBytes []byte) ([]byte, error) {
	if c == nil || c.voprf == nil || state == nil || state.finalize == nil {
		return nil, errors.New("oprf finalize state is nil")
	}
	eval, err := unmarshalVerifiableEvaluation(evalBytes)
	if err != nil {
		return nil, err
	}
	// Verifiable finalize - includes proof verification
	outputs, err := c.voprf.client.Finalize(state.finalize, eval)
	if err != nil {
		// This will fail if the server's proof is invalid (wrong key, tampered response)
		return nil, fmt.Errorf("VOPRF verification failed: %w", err)
	}
	if len(outputs) != 1 {
		return nil, errors.New("unexpected oprf output count")
	}
	return outputs[0], nil
}

// Evaluate computes the OPRF on a blinded input and generates a proof.
// In verifiable mode, the proof allows clients to verify the server used the correct key.
func (s *OPRFServer) Evaluate(requestBytes []byte) ([]byte, error) {
	if s == nil {
		return nil, errors.New("oprf server is nil")
	}
	req, err := unmarshalEvaluationRequest(requestBytes)
	if err != nil {
		return nil, err
	}
	// Lock to protect against race in CIRCL's lazy public key initialization
	s.mu.Lock()
	eval, err := s.server.Evaluate(req)
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	// Marshal the evaluation with proof for verifiability
	return marshalVerifiableEvaluation(eval)
}

// marshalVerifiableEvaluation serializes the evaluation including the DLEQ proof.
func marshalVerifiableEvaluation(eval *oprf.Evaluation) ([]byte, error) {
	elemBytes, err := marshalOPRFElements(eval.Elements)
	if err != nil {
		return nil, err
	}
	// Serialize the proof
	proofBytes, err := eval.Proof.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal VOPRF proof: %w", err)
	}
	// Format: elements || proofLen(2) || proof
	out := make([]byte, len(elemBytes)+2+len(proofBytes))
	copy(out, elemBytes)
	binary.BigEndian.PutUint16(out[len(elemBytes):], uint16(len(proofBytes)))
	copy(out[len(elemBytes)+2:], proofBytes)
	return out, nil
}

func marshalOPRFElements(elements []oprf.Blinded) ([]byte, error) {
	if len(elements) > 0xFFFF {
		return nil, errors.New("oprf element count exceeds uint16")
	}
	elemLen := int(oprfSuite.Group().Params().CompressedElementLength)
	out := make([]byte, 2, 2+len(elements)*elemLen)
	binary.BigEndian.PutUint16(out, uint16(len(elements)))
	for _, element := range elements {
		blob, err := element.MarshalBinaryCompress()
		if err != nil {
			return nil, err
		}
		out = append(out, blob...)
	}
	return out, nil
}

func unmarshalOPRFElements(data []byte) ([]oprf.Blinded, error) {
	if len(data) < 2 {
		return nil, errors.New("oprf element payload too short")
	}
	count := int(binary.BigEndian.Uint16(data[:2]))
	elemLen := int(oprfSuite.Group().Params().CompressedElementLength)
	expected := 2 + count*elemLen
	if len(data) != expected {
		return nil, errors.New("oprf element payload length mismatch")
	}
	elements := make([]oprf.Blinded, count)
	offset := 2
	for i := 0; i < count; i++ {
		element := oprfSuite.Group().NewElement()
		if err := element.UnmarshalBinary(data[offset : offset+elemLen]); err != nil {
			return nil, err
		}
		elements[i] = element
		offset += elemLen
	}
	return elements, nil
}

func unmarshalEvaluationRequest(data []byte) (*oprf.EvaluationRequest, error) {
	elements, err := unmarshalOPRFElements(data)
	if err != nil {
		return nil, err
	}
	return &oprf.EvaluationRequest{Elements: elements}, nil
}

// unmarshalVerifiableEvaluation deserializes an evaluation with DLEQ proof.
func unmarshalVerifiableEvaluation(data []byte) (*oprf.Evaluation, error) {
	if len(data) < 4 {
		return nil, errors.New("verifiable evaluation too short")
	}

	// First, parse the elements length to know where elements end
	elemCount := int(binary.BigEndian.Uint16(data[:2]))
	elemLen := int(oprfSuite.Group().Params().CompressedElementLength)
	elemTotalLen := 2 + elemCount*elemLen

	if len(data) < elemTotalLen+2 {
		return nil, errors.New("verifiable evaluation truncated")
	}

	elements, err := unmarshalOPRFElements(data[:elemTotalLen])
	if err != nil {
		return nil, err
	}

	// Parse proof length and proof
	proofLen := int(binary.BigEndian.Uint16(data[elemTotalLen:]))
	if len(data) != elemTotalLen+2+proofLen {
		return nil, errors.New("verifiable evaluation proof length mismatch")
	}

	proofBytes := data[elemTotalLen+2:]
	proof := &dleq.Proof{}
	if err := proof.UnmarshalBinary(oprfSuite.Group(), proofBytes); err != nil {
		return nil, fmt.Errorf("unmarshal VOPRF proof: %w", err)
	}

	return &oprf.Evaluation{Elements: elements, Proof: proof}, nil
}

// OPRFPrivateKey is a type alias for the underlying OPRF private key.
type OPRFPrivateKey = oprf.PrivateKey

// ComputeDirectOPRF computes the OPRF output F(key, input) directly without blinding.
// This is used by the ingest pipeline to precompute OPRF outputs for known-bad hashes.
// The output matches what a client would get from Finalize() for the same input.
func (s *OPRFServer) ComputeDirectOPRF(input []byte) ([]byte, error) {
	if s == nil {
		return nil, errors.New("oprf server is nil")
	}
	// Lock to protect against race in CIRCL's lazy public key initialization
	s.mu.Lock()
	output, err := s.server.FullEvaluate(input)
	s.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("OPRF full evaluate: %w", err)
	}
	return output, nil
}
