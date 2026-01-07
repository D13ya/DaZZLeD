package crypto

import (
	"crypto/rand"
	"encoding/binary"
	"errors"

	"github.com/cloudflare/circl/oprf"
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

// OPRFClient implements OPRFClientInterface using Ristretto255.
type OPRFClient struct {
	client oprf.Client
}

// OPRFServer implements OPRFServerInterface using Ristretto255.
type OPRFServer struct {
	server oprf.Server
}

// OPRFState holds the blinding state needed for finalization.
type OPRFState struct {
	finalize *oprf.FinalizeData
}

// NewOPRFClient creates a new OPRF client using Ristretto255.
func NewOPRFClient() *OPRFClient {
	return &OPRFClient{client: oprf.NewClient(oprfSuite)}
}

// NewOPRFServer creates a new OPRF server with the given private key.
func NewOPRFServer(privateKey *oprf.PrivateKey) *OPRFServer {
	return &OPRFServer{server: oprf.NewServer(oprfSuite, privateKey)}
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
	if c == nil {
		return nil, nil, errors.New("oprf client is nil")
	}
	fin, req, err := c.client.Blind([][]byte{input})
	if err != nil {
		return nil, nil, err
	}
	wire, err := marshalOPRFElements(req.Elements)
	if err != nil {
		return nil, nil, err
	}
	return &OPRFState{finalize: fin}, wire, nil
}

// Finalize unblinds the server response to get the OPRF output.
func (c *OPRFClient) Finalize(state *OPRFState, evalBytes []byte) ([]byte, error) {
	if c == nil || state == nil || state.finalize == nil {
		return nil, errors.New("oprf finalize state is nil")
	}
	eval, err := unmarshalEvaluation(evalBytes)
	if err != nil {
		return nil, err
	}
	outputs, err := c.client.Finalize(state.finalize, eval)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.New("unexpected oprf output count")
	}
	return outputs[0], nil
}

// Evaluate computes the OPRF on a blinded input.
func (s *OPRFServer) Evaluate(requestBytes []byte) ([]byte, error) {
	if s == nil {
		return nil, errors.New("oprf server is nil")
	}
	req, err := unmarshalEvaluationRequest(requestBytes)
	if err != nil {
		return nil, err
	}
	eval, err := s.server.Evaluate(req)
	if err != nil {
		return nil, err
	}
	return marshalOPRFElements(eval.Elements)
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

func unmarshalEvaluation(data []byte) (*oprf.Evaluation, error) {
	elements, err := unmarshalOPRFElements(data)
	if err != nil {
		return nil, err
	}
	return &oprf.Evaluation{Elements: elements, Proof: nil}, nil
}
