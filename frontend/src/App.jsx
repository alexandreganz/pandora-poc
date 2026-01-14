import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function ProductCard({ product, onAddToCart, tryOnImage }) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // Build image array: product images + try-on image if available
  const images = [product.image1, product.image2]
  if (tryOnImage) {
    images.push(tryOnImage)
  }

  const goToNext = () => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length)
  }

  const goToPrev = () => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length)
  }

  return (
    <div className="product-card">
      <div className="product-image-container">
        <img
          src={images[currentImageIndex]}
          alt={currentImageIndex === 2 ? `${product.name} - Your Try-On` : product.name}
          className="product-image"
        />

        {images.length > 1 && (
          <>
            <button className="carousel-btn carousel-btn-prev" onClick={goToPrev}>
              ‹
            </button>
            <button className="carousel-btn carousel-btn-next" onClick={goToNext}>
              ›
            </button>
            <div className="carousel-dots">
              {images.map((_, index) => (
                <button
                  key={index}
                  className={`carousel-dot ${index === currentImageIndex ? 'active' : ''}`}
                  onClick={() => setCurrentImageIndex(index)}
                />
              ))}
            </div>
          </>
        )}
      </div>
      <div className="product-info">
        <h3 className="product-name">{product.name}</h3>
        <p className="product-price">${product.price.toFixed(2)}</p>
        <button
          className="try-on-btn"
          onClick={() => onAddToCart(product)}
        >
          Add to Cart
        </button>
      </div>
    </div>
  )
}

function TryOnModal({ products, onClose, onTryOnComplete }) {
  const [frontPhoto, setFrontPhoto] = useState(null)
  const [frontPreview, setFrontPreview] = useState(null)
  const [photosUploaded, setPhotosUploaded] = useState(false)
  const [selectedEarring, setSelectedEarring] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingAll, setLoadingAll] = useState(false)
  const [results, setResults] = useState({})
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setFrontPhoto(file)
        setFrontPreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePhotosSubmit = (e) => {
    e.preventDefault()
    if (frontPhoto) {
      setPhotosUploaded(true)
    }
  }

  const handleTryOn = async (product) => {
    if (results[product.id]) {
      setSelectedEarring(product.id)
      return
    }

    setLoading(true)
    setSelectedEarring(product.id)
    setError(null)

    const formData = new FormData()
    formData.append('front_photo', frontPhoto)
    formData.append('earring_id', product.id)

    try {
      const response = await fetch(`${API_URL}/api/try-on`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Failed to process images')
      }

      const data = await response.json()
      setResults(prev => ({ ...prev, [product.id]: data }))

      // Pass the result to parent App component
      if (onTryOnComplete) {
        onTryOnComplete(product.id, data.image)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleTryOnAll = async () => {
    console.log('Starting Try All At Once...')
    setLoadingAll(true)
    setError(null)
    setSelectedEarring(null)

    const formData = new FormData()
    formData.append('front_photo', frontPhoto)

    try {
      console.log('Sending request to backend...')
      const response = await fetch(`${API_URL}/api/try-on-all`, {
        method: 'POST',
        body: formData,
      })

      console.log('Response received:', response.status)

      if (!response.ok) {
        const err = await response.json()
        console.error('API Error:', err)
        throw new Error(err.detail || 'Failed to process images')
      }

      const data = await response.json()
      console.log('Processing results:', data)

      // Process results array and update state
      if (data.results && Array.isArray(data.results)) {
        const newResults = {}

        data.results.forEach(result => {
          if (result.success) {
            newResults[result.product_id] = { image: result.image }

            // Pass each result to parent App component
            if (onTryOnComplete) {
              onTryOnComplete(result.product_id, result.image)
            }
          }
        })

        setResults(prev => ({ ...prev, ...newResults }))

        // Show success message
        const successCount = data.results.filter(r => r.success).length
        console.log(`Successfully generated ${successCount} out of ${products.length} try-ons`)

        if (successCount === products.length) {
          // All successful - turn off loading to show success indicator
          setLoadingAll(false)
          setError(null)

          // Auto-close modal after 2 seconds
          setTimeout(() => {
            console.log('Closing modal...')
            onClose()
          }, 2000)
        } else if (successCount > 0) {
          // Partial success
          setLoadingAll(false)
          setError(`Generated ${successCount} out of ${products.length} try-on images`)
        } else {
          // All failed
          setLoadingAll(false)
          setError('Failed to generate any try-on images')
        }
      } else {
        console.error('Invalid response format:', data)
        setLoadingAll(false)
        setError('Invalid response from server')
      }
    } catch (err) {
      console.error('Try-On All Error:', err)
      setError(err.message)
      setLoadingAll(false)
    }
  }

  const resetPhotos = () => {
    setPhotosUploaded(false)
    setSelectedEarring(null)
    setResults({})
    setError(null)
  }

  return (
    <div className="modal-overlay" onClick={loadingAll ? undefined : onClose}>
      <div className="modal try-on-modal" onClick={(e) => e.stopPropagation()}>
        {!loadingAll && (
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        )}
        <h2 className="modal-title">Virtual Try-On</h2>

        {!photosUploaded ? (
          <form className="try-on-form" onSubmit={handlePhotosSubmit}>
            <p className="try-on-intro">Upload a front-facing photo to try on any earring from our collection.</p>

            <div className="upload-group">
              <label className="upload-label">Front Photo</label>
              <p className="upload-hint">Face the camera directly with both ears visible</p>
              <div
                className="upload-input"
                onClick={() => document.getElementById('front-input').click()}
              >
                <input
                  id="front-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
                {frontPreview ? (
                  <img src={frontPreview} alt="Front preview" className="preview-image" />
                ) : (
                  <span>Click to upload your photo</span>
                )}
              </div>
            </div>

            <button
              type="submit"
              className="submit-btn"
              disabled={!frontPhoto}
            >
              Continue to Try-On
            </button>
          </form>
        ) : (
          <div className="try-on-selection">
            <div className="photos-preview">
              <img src={frontPreview} alt="Your photo" />
              {!loadingAll && (
                <button className="change-photos-btn" onClick={resetPhotos}>
                  Change Photo
                </button>
              )}
            </div>

            {!loadingAll && (
              <>
                <h3 className="selection-title">Select an earring to try on:</h3>

                <div className="try-all-container">
                  <button
                    className="try-all-btn"
                    onClick={handleTryOnAll}
                    disabled={loading}
                  >
                    Try All At Once
                  </button>
                  <p className="try-all-hint">Generate try-on photos for all 5 earrings in one go</p>
                </div>

                <div className="selection-divider">
                  <span>OR</span>
                </div>

                <div className="earring-selection-grid">
                  {products.map((product) => (
                    <div
                      key={product.id}
                      className={`earring-option ${selectedEarring === product.id ? 'selected' : ''} ${results[product.id] ? 'has-result' : ''}`}
                      onClick={() => handleTryOn(product)}
                    >
                      <img src={product.image1} alt={product.name} />
                      <span>{product.name}</span>
                      {results[product.id] && <span className="check-mark">✓</span>}
                    </div>
                  ))}
                </div>
              </>
            )}

            {error && <p className="error-message">{error}</p>}

            {loadingAll && (
              <div className="loading-indicator batch-loading">
                <div className="loading-spinner-large"></div>
                <h3>Generating Your Try-Ons</h3>
                <p className="loading-progress">Processing all 5 earrings...</p>
                <div className="loading-steps">
                  <div className="loading-step">✓ Creating pose variation</div>
                  <div className="loading-step active">⟳ Applying earrings</div>
                  <div className="loading-step">◯ Finalizing images</div>
                </div>
                <p className="loading-subtext">This may take 30-60 seconds. Please wait...</p>
              </div>
            )}

            {loading && !loadingAll && (
              <div className="loading-indicator">
                Processing your try-on...
              </div>
            )}

            {!loadingAll && !loading && Object.keys(results).length === products.length && (
              <div className="success-indicator">
                <div className="success-icon">✓</div>
                <h3>All Try-Ons Generated!</h3>
                <p>Closing modal and updating carousels...</p>
              </div>
            )}

            {selectedEarring && results[selectedEarring] && (
              <div className="result-container">
                <h3>Try-On Result: {products.find(p => p.id === selectedEarring)?.name}</h3>
                <img src={results[selectedEarring].image} alt="Try-on result" className="result-image" />
                <div className="result-analysis">
                  <p><strong>Analysis:</strong></p>
                  <p>Earring size: {results[selectedEarring].analysis.earring_size_px.width}x{results[selectedEarring].analysis.earring_size_px.height}px</p>
                  <p>Pixels per mm: {results[selectedEarring].analysis.pixels_per_mm.toFixed(2)}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function CheckoutModal({ cart, onClose, onClearCart }) {
  const [submitted, setSubmitted] = useState(false)

  const total = cart.reduce((sum, item) => sum + item.price, 0)

  const handleCheckout = () => {
    setSubmitted(true)
    setTimeout(() => {
      onClearCart()
      onClose()
    }, 2000)
  }

  if (submitted) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <div className="checkout-success">
            <h3>Thank you for your order!</h3>
            <p>This is a mock checkout - no payment was processed.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          &times;
        </button>
        <h2 className="modal-title">Your Cart</h2>

        {cart.length === 0 ? (
          <p style={{ textAlign: 'center' }}>Your cart is empty</p>
        ) : (
          <>
            <div className="checkout-items">
              {cart.map((item, index) => (
                <div key={index} className="checkout-item">
                  <span>{item.name}</span>
                  <span>${item.price.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="checkout-total">
              <span>Total</span>
              <span>${total.toFixed(2)}</span>
            </div>
            <button className="submit-btn" onClick={handleCheckout}>
              Complete Purchase (Mock)
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function App() {
  const [products, setProducts] = useState([])
  const [loading, setLoading] = useState(true)
  const [showTryOn, setShowTryOn] = useState(false)
  const [showCheckout, setShowCheckout] = useState(false)
  const [cart, setCart] = useState([])
  const [tryOnResults, setTryOnResults] = useState({})

  useEffect(() => {
    fetch(`${API_URL}/api/products`)
      .then((res) => res.json())
      .then((data) => {
        setProducts(data.products)
        setLoading(false)
      })
      .catch(() => {
        // Use mock data if backend is not available
        setProducts([
          {
            id: 1,
            name: 'Classic Circle',
            price: 89.0,
            dimensions: { width: 15, height: 15 },
            image1: '/images/classic-circle-1.jpg',
            image2: '/images/classic-circle-2.jpg',
          },
          {
            id: 2,
            name: 'Silver Flower',
            price: 95.0,
            dimensions: { width: 12, height: 18 },
            image1: '/images/silver-flower-1.jpg',
            image2: '/images/silver-flower-2.jpg',
          },
          {
            id: 3,
            name: 'Red Heart',
            price: 79.0,
            dimensions: { width: 10, height: 12 },
            image1: '/images/red-heart-1.jpg',
            image2: '/images/red-heart-2.jpg',
          },
          {
            id: 4,
            name: 'Gold Heart',
            price: 125.0,
            dimensions: { width: 10, height: 12 },
            image1: '/images/gold-heart-1.jpg',
            image2: '/images/gold-heart-2.jpg',
          },
          {
            id: 5,
            name: 'Silver Heart',
            price: 85.0,
            dimensions: { width: 10, height: 12 },
            image1: '/images/silver-heart-1.jpg',
            image2: '/images/silver-heart-2.jpg',
          },
          {
            id: 6,
            name: 'Blue Butterfly',
            price: 110.0,
            dimensions: { width: 22, height: 20 },
            image1: '/images/blue-butterfly-1.jpg',
            image2: '/images/blue-butterfly-2.jpg',
          },
        ])
        setLoading(false)
      })
  }, [])

  const addToCart = (product) => {
    setCart([...cart, product])
  }

  const clearCart = () => {
    setCart([])
  }

  const handleTryOnComplete = (productId, tryOnImage) => {
    setTryOnResults(prev => ({ ...prev, [productId]: tryOnImage }))
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">Pandora</div>
        <nav className="nav-links">
          <a href="#">Earrings</a>
          <a href="#">New In</a>
          <a href="#">Sale</a>
        </nav>
        <div className="header-actions">
          <button className="try-on-header-btn" onClick={() => setShowTryOn(true)}>
            Virtual Try-On
          </button>
          <div className="cart-icon" onClick={() => setShowCheckout(true)}>
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z" />
              <line x1="3" y1="6" x2="21" y2="6" />
              <path d="M16 10a4 4 0 0 1-8 0" />
            </svg>
            {cart.length > 0 && <span className="cart-count">{cart.length}</span>}
          </div>
        </div>
      </header>

      <main className="main">
        <h1 className="page-title">Earrings</h1>

        {loading ? (
          <div className="loading">Loading products...</div>
        ) : (
          <div className="product-grid">
            {products.map((product) => (
              <ProductCard
                key={product.id}
                product={product}
                onAddToCart={addToCart}
                tryOnImage={tryOnResults[product.id]}
              />
            ))}
          </div>
        )}
      </main>

      {showTryOn && (
        <TryOnModal
          products={products}
          onClose={() => setShowTryOn(false)}
          onTryOnComplete={handleTryOnComplete}
        />
      )}

      {showCheckout && (
        <CheckoutModal
          cart={cart}
          onClose={() => setShowCheckout(false)}
          onClearCart={clearCart}
        />
      )}
    </div>
  )
}

export default App
